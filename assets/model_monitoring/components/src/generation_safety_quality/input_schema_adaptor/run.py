# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GSQ Input Schema Adaptor Spark Component."""

import argparse
import json

from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import from_json
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
    try_read_mltable_in_spark_with_error,
)
from shared_utilities.df_utils import try_get_df_column


GENAI_ROOT_SPAN_SCHEMA_COLUMN = "root_span"
GENAI_TRACE_ID_SCHEMA_COLUMN = "trace_id"


def _get_input_schema_adaptor_map() -> dict:
    """Map the gsq input schema names to expected agg trace log schema colum/field name."""
    map = {
        "question": "input.prompt",
        "answer": "output.output",
        "context": "input.context",
        "ground_truth": "input.groundtruth",
    }
    return map


def _adapt_input_data_schema(df: DataFrame) -> DataFrame:
    """Adapt the input dataframe schema to fit GSQ input schema."""
    df_field_names = df.schema.fieldNames()

    # check if we need to adapt the schema
    if GENAI_ROOT_SPAN_SCHEMA_COLUMN not in df_field_names and GENAI_TRACE_ID_SCHEMA_COLUMN not in df_field_names:
        return df

    spark = init_spark()
    input_schema = spark.read.json(df.rdd.map(lambda row: row.input)).schema
    output_schema = spark.read.json(df.rdd.map(lambda row: row.output)).schema

    transformed_df = df.withColumns({
        'temp_input': from_json(df.input, input_schema),
        'temp_output': from_json(df.output, output_schema),
    }).select('trace_id', 'temp_input.*', 'temp_output.*')

    # filter down to gsq schema
    transformed_df = transformed_df.withColumns(
        {
            col_name: df_column for col_name, column_mapping in _get_input_schema_adaptor_map().items()
            if (df_column := try_get_df_column(transformed_df, column_mapping)) is not None
        }
    )
    return transformed_df


def run():
    """Adapt the production dataset schema to match GSQ input schema if necessary."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--adapted_production_data", type=str, required=True)

    args = parser.parse_args()

    production_data_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    transformed_df = _adapt_input_data_schema(production_data_df)

    save_spark_df_as_mltable(transformed_df, args.adapted_production_data)


if __name__ == "__main__":
    run()
