# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Groundedness Metrics Computing Spark Component."""

import argparse
from pyspark.sql.functions import col, udf, sum
from pyspark.sql.types import IntegerType
from shared_utilities.io_utils import (
    read_mltable_in_spark,
    save_spark_df_as_mltable,
    init_spark,
)


def _calculate_passrate(df, threshold):
    df_with_buckets = df.filter(
        col("metric_name").contains("GroundednessCount_")
    ).withColumn(
        "bucket",
        udf(lambda metric_name: int(metric_name.split("_")[1]), IntegerType())(
            col("metric_name")
        ),
    )
    passing = (
        df_with_buckets.filter(col("bucket") >= threshold)
        .select(sum("metric_value"))
        .head()[0]
    )
    total = df_with_buckets.select(sum("metric_value")).head()[0]
    return passing / total


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundedness_histogram", type=str)
    parser.add_argument("--groundedness_threshold", type=int, required=True)
    parser.add_argument("--signal_metrics", type=str)
    args = parser.parse_args()

    threshold = args.groundedness_threshold

    histogram_df = read_mltable_in_spark(args.groundedness_histogram)

    # add total row count
    spark = init_spark()

    histogram_df = histogram_df.withColumn(
        "metric_value", histogram_df["metric_value"].cast("float")
    )
    metrics_df = histogram_df.union(
        spark.createDataFrame(
            [
                (
                    "AggregatedGroundednessPassRate",
                    _calculate_passrate(histogram_df, threshold),
                    "numerical",
                    "",
                )
            ],
            histogram_df.schema,
        )
    )
    save_spark_df_as_mltable(metrics_df, args.signal_metrics)


if __name__ == "__main__":
    run()
