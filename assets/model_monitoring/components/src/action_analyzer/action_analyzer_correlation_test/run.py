# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer correlation test."""

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
            StructField("CorrelationTest", StringType(), True),
        ]
    )
    # Create a new DataFrame with the metadata
    df = spark.createDataFrame([], metadata_schema)
    save_spark_df_as_mltable(df, output_path)


def run():
    """Correlation test."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--action_data", type=str)
    parser.add_argument("--data_with_action_metric_score", type=str)
    args = parser.parse_args()

    data_with_action_metric_score_df = try_read_mltable_in_spark_with_error(
        args.data_with_groups, "data_with_action_metric_score"
    )

    if data_with_action_metric_score_df.isEmpty():
        print("empty metrics score data")

    print("correlation test")

    write_empty_dataframe(args.action_data)


if __name__ == "__main__":
    run()
