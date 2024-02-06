# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer output actions."""

import argparse
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)
from shared_utilities.io_utils import try_read_mltable_in_spark_with_error, save_spark_df_as_mltable, init_spark


def write_empty_dataframe(output_path):
    """Write an empty action data frame."""
    spark = init_spark()
    metadata_schema = StructType(
        [
            StructField("OutputAction", StringType(), True),
        ]
    )
    # Create a new DataFrame with the metadata
    df = spark.createDataFrame([], metadata_schema)
    save_spark_df_as_mltable(df, output_path)


def run():
    """Merge and output actions."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_output", type=str)
    parser.add_argument("--action_data", type=str)
    args = parser.parse_args()

    action_data_df = try_read_mltable_in_spark_with_error(
        args.data_with_groups, "action_data"
    )

    if action_data_df.isEmpty():
        print("empty action data")

    print("output action")

    write_empty_dataframe(args.action_output)


if __name__ == "__main__":
    run()
