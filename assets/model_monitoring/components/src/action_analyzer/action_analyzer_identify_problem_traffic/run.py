# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer identify problem traffic."""

import argparse
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)
from shared_utilities.io_utils import save_spark_df_as_mltable, init_spark


def write_empty_dataframe(output_path):
    """Write an empty action data frame."""
    spark = init_spark()
    metadata_schema = StructType(
        [
            StructField("IdentifyProblemTraffic", StringType(), True),
        ]
    )
    # Create a new DataFrame with the metadata
    df = spark.createDataFrame([], metadata_schema)
    save_spark_df_as_mltable(df, output_path)


def run():
    """Identify problem traffic."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_groups", type=str)
    parser.add_argument("--production_dataset", type=str)
    parser.add_argument("--signal_scored_output", type=str)
    parser.add_argument("--violated_metric_names", type=str)
    args = parser.parse_args()

    if args.violated_metric_names:
        print("No violated metrics, creating an empty action dataframe.")
        write_empty_dataframe(args.data_with_groups)
        return

    write_empty_dataframe(args.data_with_groups)


if __name__ == "__main__":
    run()
