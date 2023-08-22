# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Annotation Metrics Computing Spark Component."""

import argparse
from pyspark.sql.functions import col, udf, sum
from pyspark.sql.types import IntegerType
from shared_utilities.io_utils import (
    read_mltable_in_spark,
    save_spark_df_as_mltable,
    init_spark,
)


def _calculate_passrate(df, threshold, metric_name):
    metric_prefix = f"{metric_name}Count_"
    df_with_buckets = df.filter(
        col("metric_name").contains(metric_prefix)
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
    if total == 0:
        return 1
    return passing / total


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_names", type=str)
    parser.add_argument("--annotation_histogram", type=str)
    parser.add_argument("--annotation_rating_threshold", type=int, required=True)
    parser.add_argument("--signal_metrics", type=str)
    args = parser.parse_args()

    threshold = args.annotation_rating_threshold
    metric_names = [m.strip() for m in args.metric_names.split(",")]
    histogram_df = read_mltable_in_spark(args.annotation_histogram)
    spark = init_spark()
    # Cast to float because metric_value was integer so far
    # but we're adding percentages now.
    histogram_df = histogram_df.withColumn(
        "metric_value", histogram_df["metric_value"].cast("float")
    )

    aggregated_metrics_df = histogram_df
    for metric_name in metric_names:
        compact_metric_name = metric_name.replace(" ", "").title()
        print(compact_metric_name)
        metric_df = spark.createDataFrame(
                [
                    (
                        f"Aggregated{compact_metric_name}PassRate",
                        _calculate_passrate(histogram_df, threshold, compact_metric_name),
                        "numerical",
                        "",
                    )
                ],
                histogram_df.schema,
            )
        aggregated_metrics_df = aggregated_metrics_df.union(metric_df)
    save_spark_df_as_mltable(aggregated_metrics_df, args.signal_metrics)


if __name__ == "__main__":
    run()
