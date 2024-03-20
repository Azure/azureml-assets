# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GSQ Input Schema Adaptor Spark Component."""

import argparse
from copy import copy
import json

from model_data_collector_preprocessor.spark_run import _convert_complex_columns_to_json_string
from pyspark.sql import DataFrame, Row
from pyspark.sql.functions import from_json, max
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
    try_read_mltable_in_spark_with_error,
)
from shared_utilities.momo_exceptions import InvalidInputError
from shared_utilities.df_utils import has_duplicated_columns, try_get_df_column
from shared_utilities.constants import (
    GENAI_ROOT_SPAN_SCHEMA_COLUMN,
    GENAI_TRACE_ID_SCHEMA_COLUMN,
    GSQ_PROMPT_COLUMN,
    GSQ_COMPLETION_COLUMN,
    GSQ_CONTEXT_COLUMN,
    GSQ_GROUND_TRUTH_COLUMN,
)


# def check_row_has_gsq_schema(row: dict, expected_schema: list) -> list:
#     """Check each dataframe row for expected schema value."""
#     input_dict: dict = json.loads(row.get("input", "null"))
#     output_dict: dict = json.loads(row.get("output", "null"))
#     if input_dict is None:
#         print(f"Row entry with trace id = {row.get('trace_id', '')} has no input field.")
#         input_dict = {}
#     if output_dict is None:
#         print(f"Row entry with trace id = {row.get('trace_id', '')} has no output field.")
#         output_dict = {}
#     entry = []
#     for column in expected_schema:
#         if column in input_dict:
#             entry.append(input_dict.get(column))
#         elif column in output_dict:
#             entry.append(output_dict.get(column))
#         else:
#             entry.append(None)
#     return [tuple(entry)]


def expand_json_to_gsq_schema(row, gsq_columns: list):
    """Expand the json GSQ schema."""
    input_json_data = json.loads(row.input)
    output_json_data = json.loads(row.output)
    print(f"input: {input_json_data}")
    print(f"output: {output_json_data}")
    output_dict = {}
    for col in gsq_columns:
        if col in input_json_data:
            output_dict[col] = input_json_data[col]
        elif col in output_json_data:
            output_dict[col] = output_json_data[col]
        else:
            output_dict[col] = None
    return [Row(**output_dict)]


def _adapt_input_data_schema(
        df: DataFrame,
        prompt_column_name,
        completion_column_name,
        context_column_name,
        ground_truth_column_name) -> DataFrame:
    """Adapt the input dataframe schema to fit GSQ input schema."""
    df_field_names = df.schema.fieldNames()

    # check if we need to adapt the schema
    if GENAI_ROOT_SPAN_SCHEMA_COLUMN not in df_field_names and GENAI_TRACE_ID_SCHEMA_COLUMN not in df_field_names:
        print(
            f"Did not find '{GENAI_ROOT_SPAN_SCHEMA_COLUMN}' and '{GENAI_TRACE_ID_SCHEMA_COLUMN}' columns in dataset. "
            "skip adapting dataframe logic."
        )
        return df

    print("Adapting the GenAI production data to gsq input schema...")
    # spark = init_spark()
    # columns_to_select = [
    #     prompt_column_name, completion_column_name, context_column_name, ground_truth_column_name,
    #     GENAI_TRACE_ID_SCHEMA_COLUMN, GENAI_ROOT_SPAN_SCHEMA_COLUMN
    # ]
    # transormed_df = df.rdd.map(lambda x: x.asDict()).flatMap(lambda y: check_row_has_gsq_schema(y, columns_to_select)).toDF(columns_to_select)

    # columns_to_check = [prompt_column_name, completion_column_name, context_column_name, ground_truth_column_name]
    # rows_with_data = transormed_df.select(*columns_to_check).agg(*[max(c).alias(c) for c in columns_to_check]).take(1)[0]
    # transformed_df = transormed_df.select(*[c for c in df.columns if rows_with_data[c] is not None])
    # transformed_df.show()
    # transformed_df.printSchema()

    # UDF
    spark = init_spark()
    gsq_schema = [prompt_column_name, completion_column_name, context_column_name, ground_truth_column_name]
    broadcast_schema = spark.sparkContext.broadcast(gsq_schema)

    transformed_df = df.rdd.flatMap(lambda row: expand_json_to_gsq_schema(row, broadcast_schema.value)).toDF()
    transformed_df.show()

    broadcast_schema.destroy()

    # drop rows with all None
    # transformed_df = transformed_df.dropna(how="all", subset=gsq_schema)
    # # drop columns with all None
    # rows_with_data = transformed_df.select(*columns_to_select).agg(*[max(c).alias(c) for c in columns_to_select]).take(1)[0]
    # transformed_df = transformed_df.select(*[c for c in df.columns if rows_with_data[c] is not None])
    # transformed_df.show()
    # transformed_df.printSchema()

    # transformed_df.show()
    # transformed_df.printSchema()

    drop_rate = transformed_df.count() / df.count()
    if drop_rate > 0.10 and drop_rate < 0.30:
        print("first warning. Missing partial data.")
    elif drop_rate > 0.30 and drop_rate < 0.67:
        print("Second warning. Missing substantial amount of data.")
    elif drop_rate > 0.67:
        raise InvalidInputError("Job missing too much data.")

    if has_duplicated_columns(df):
        raise InvalidInputError(
            "Expanding the input and output columms resulted in duplicate columns."
            f" The dataframe's columns are: {transformed_df.columns}."
            " This scenario is unsupported as of right now. Please clean up the production data logs"
            " so there are no duplicate fields in 'input' and 'output' columns."
        )

    # flatten unpacked json columns to json_string if necessary
    transformed_df = _convert_complex_columns_to_json_string(transformed_df)

    return transformed_df


def run():
    """Adapt the production dataset schema to match GSQ input schema if necessary."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--adapted_production_data", type=str, required=True)

    parser.add_argument("--prompt_column_name", type=str, default=GSQ_PROMPT_COLUMN)
    parser.add_argument("--completion_column_name", type=str, default=GSQ_COMPLETION_COLUMN)
    parser.add_argument("--context_column_name", type=str, default=GSQ_CONTEXT_COLUMN)
    parser.add_argument("--ground_truth_column_name", type=str, default=GSQ_GROUND_TRUTH_COLUMN)

    args = parser.parse_args()

    production_data_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    adapted_df = _adapt_input_data_schema(
        production_data_df,
        args.prompt_column_name,
        args.completion_column_name,
        args.context_column_name,
        args.ground_truth_column_name,
    )

    print("df adapted from production data:")
    adapted_df.show()
    adapted_df.printSchema()

    save_spark_df_as_mltable(adapted_df, args.adapted_production_data)


if __name__ == "__main__":
    run()
