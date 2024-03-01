# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GSQ Input Schema Adaptor Spark Component."""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.functions import from_json
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
    try_read_mltable_in_spark_with_error,
)
from shared_utilities.momo_exceptions import InvalidInputError
from model_data_collector_preprocessor.spark_run import _convert_complex_columns_to_json_string
from shared_utilities.df_utils import has_duplicated_columns
from shared_utilities.constants import GENAI_ROOT_SPAN_SCHEMA_COLUMN, GENAI_TRACE_ID_SCHEMA_COLUMN


def _adapt_input_data_schema(df: DataFrame) -> DataFrame:
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
    spark = init_spark()
    try:
        sampled_df_slice = df.sample(0.2)
        if sampled_df_slice.isEmpty():
            print(
                "Not enough data resulting from production data and sample rate. "
                "Using first 5 rows of production data instead."
            )
            sampled_df_slice = df.limit(5)
        print("Sampled data used to get 'input'/'output' json schema:")
        sampled_df_slice.show()
        sampled_df_slice.printSchema()

        input_schema = spark.read.json(sampled_df_slice.rdd.map(lambda row: row.input), mode="FAILFAST").schema
        output_schema = spark.read.json(sampled_df_slice.rdd.map(lambda row: row.output), mode="FAILFAST").schema
    except Exception as ex:
        if "Malformed records are detected in schema inference. Parse Mode: FAILFAST." in str(ex):
            raise InvalidInputError(
                "Failed to parse the input/output column json string for the trace logs provided."
                " The input/output columns in the inputted production data are not in a parseable"
                " json string format. Please double-check the data columns are being passed or"
                " stored correctly."
            )
        raise ex

    df = df.withColumns({
        'temp_input': from_json(df.input, input_schema),
        'temp_output': from_json(df.output, output_schema),
    })

    print("Data with expanded from_json():")
    df.show()
    df.printSchema()

    df = df.select('trace_id', 'temp_input.*', 'temp_output.*', 'root_span')

    if has_duplicated_columns(df):
        raise InvalidInputError(
            "Expanding the input and output columms resulted in duplicate columns."
            " This scenario is unsupported as of right now please clean up the production data logs"
            " so there are no duplicate fields in 'input' and 'output' columns."
        )

    # flatten unpacked json columns to json_string if necessary
    df = _convert_complex_columns_to_json_string(df)

    return df


def run():
    """Adapt the production dataset schema to match GSQ input schema if necessary."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--adapted_production_data", type=str, required=True)

    args = parser.parse_args()

    production_data_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    adapted_df = _adapt_input_data_schema(production_data_df)

    print("df adapted from production data:")
    adapted_df.show()
    adapted_df.printSchema()

    save_spark_df_as_mltable(adapted_df, args.adapted_production_data)


if __name__ == "__main__":
    run()
