# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
from pyspark.sql.functions import split, trim, col
from shared_utilities.io_utils import (
    save_spark_df_as_mltable,
    try_read_mltable_in_spark,
    try_read_mltable_in_spark_with_warning,
)
from shared_utilities.df_utils import select_columns_from_spark_df
from compute_data_quality_metrics import compute_data_quality_metrics


def _filter_metrics(df, metrics, data_type):
    return df.filter(
        (col("metricName").isin(metrics) & (col("dataType") == data_type))
        | (col("dataType") != data_type)
    )


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--data_statistics", type=str)
    parser.add_argument("--feature_names", type=str, default=None)
    parser.add_argument("--signal_metrics", type=str)
    parser.add_argument("--categorical_metrics", type=str)
    parser.add_argument("--numerical_metrics", type=str)
    args = parser.parse_args()

    # READ INPUT TABLES
    df = try_read_mltable_in_spark_with_warning(args.input_data, "input_data")
    data_stats_table = try_read_mltable_in_spark_with_warning(
        args.data_statistics, "data_statistics"
    )

    if not df or not data_stats_table:
        print("Skipping data quality metrics computation.")
        return

    # Select columns
    if args.feature_names:
        features = try_read_mltable_in_spark(args.feature_names, "feature_names")

        if not features:
            print("Skipping data quality metrics computation.")
            return

        select_columns = [row["featureName"] for row in features.collect()]
        df = select_columns_from_spark_df(df, select_columns)

    # CONVERT "set" COLUMN BACK TO ARRAY TYPE STRING
    data_stats_table = data_stats_table.withColumn(
        "set", split(trim("set"), "( +)?, ?")
    )

    violation_df = compute_data_quality_metrics(df, data_stats_table)
    violation_df = _filter_metrics(
        violation_df, args.categorical_metrics.split(","), data_type="Categorical"
    )
    violation_df = _filter_metrics(
        violation_df, args.numerical_metrics.split(","), data_type="Numerical"
    )
    # Save metrics in default blob store and log it in active run
    save_spark_df_as_mltable(violation_df, args.signal_metrics)


if __name__ == "__main__":
    run()
