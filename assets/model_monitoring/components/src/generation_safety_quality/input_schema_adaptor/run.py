# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GSQ Input Schema Adaptor Spark Component."""

import argparse
import json

from pyspark.sql import DataFrame
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import lit, posexplode, from_json
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
    try_read_mltable_in_spark_with_error,
)


GENAI_ROOT_SPAN_SCHEMA_COLUMN = "root_span"
GENAI_TRACE_ID_SCHEMA_COLUMN = "trace_id"
GENAI_INPUT_COLUMN = "input"
GENAI_OUTPUT_COLUMN = "output"


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
    data_schema_field_names = df.schema.fieldNames()

    # check if we need to adapt the schema
    if GENAI_ROOT_SPAN_SCHEMA_COLUMN not in data_schema_field_names and GENAI_TRACE_ID_SCHEMA_COLUMN not in data_schema_field_names:
        return df

    spark = init_spark()
    input_json_schema = spark.read.json(df.select(GENAI_INPUT_COLUMN).rdd.map(lambda row: row.input)).schema
    output_json_schema = spark.read.json(df.select(GENAI_OUTPUT_COLUMN).rdd.map(lambda row: row.output)).schema
    df.withColumns(
        {
            'temp_input_json_col': from_json(df.input, input_json_schema),
            'temp_output_json_col': from_json(df.output, output_json_schema),
        }
    )

    for key, map_name in _get_input_schema_adaptor_map().items():
        col_name, field = map_name.split('.')
        if col_name == GENAI_INPUT_COLUMN:
            pass

    return df


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
