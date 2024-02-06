# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Action Analyzer metric calculation."""

import argparse
from shared_utilities.io_utils import try_read_mltable_in_spark_with_error, save_spark_df_as_mltable, init_spark


def write_empty_dataframe():
    """Write an empty action data frame."""
    spark = init_spark()
    metadata_schema = StructType(
        [
            StructField("MetricsCalculation", StringType(), True),
        ]
    )
    # Create a new DataFrame with the metadata
    df = spark.createDataFrame([], metadata_schema)
    save_spark_df_as_mltable(df, args.signal_metrics)


def run():
    """Calculate metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_with_action_metric_score", type=str)
    parser.add_argument("--data_with_groups", type=str)
    args = parser.parse_args()

    data_with_groups_df = try_read_mltable_in_spark_with_error(
        args.data_with_groups, "data_with_groups"
    )

    if data_with_groups_df.isEmpty():
        print("empty groups data")

    print("calculate metrics")

    write_empty_dataframe()


if __name__ == "__main__":
    run()
