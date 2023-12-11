# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Annotation Metrics Computing Spark Component."""

import argparse
from pyspark.sql.functions import lit
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark,
)
from compute_token_statistics_metrics import impute_ids_for_failed_calls, \
    check_data_quality, compute_GPU_utilization_metrics, compute_GPU_waste_metrics


def run():
    """Execute the main function."""
    # Init spark session
    spark = init_spark()

    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_dataset", type=str)
    parser.add_argument("--group_pivot_column_name", type=str, required=False, nargs="?")
    parser.add_argument("--signal_metrics", type=str)

    args = parser.parse_args()
    token_df = try_read_mltable_in_spark_with_error(args.token_dataset)
    group_pivot_column_name = args.group_pivot_column_name

    # failed calls dont have an id, so imputing them
    token_df = impute_ids_for_failed_calls(token_df)

    # basic data quality checks and filter out rows that dont meet the quality criteria
    token_df = check_data_quality(token_df)

    # designate the group column
    token_df = token_df.withColumnRenamed("node_id", "group")

    #  designate the group_pivot column
    #  if group_pivot_column_name is null, then add a column with value Not Provided
    if group_pivot_column_name is None:
        token_df = token_df.withColumn("group_pivot", lit('').cast("string"))
    else:
        token_df = token_df.withColumnRenamed(group_pivot_column_name, "group_pivot")

    # total tokens = prompt tokens + completion tokens
    token_df = token_df.withColumn("total_tokens",
                                   token_df["prompt_tokens"] + token_df["completion_tokens"])

    # compute GPU utilization metrics for group
    # create a copy of token_df where group_pivot is set to Aggregate
    token_df_aggregate_group_pivot = token_df.withColumn("group_pivot", lit('Aggregate').cast("string"))
    # compute GPU utilization metrics for group
    gpu_utilization_metrics_group_only = compute_GPU_utilization_metrics(token_df_aggregate_group_pivot)
    # compute GPU waste metrics for group
    # These metrics are computed if we have max_tokens and finish_reason in the dataset
    gpu_waste_metrics_group_only = spark.createDataFrame([], gpu_utilization_metrics_group_only.schema)
    if ("finish_reason" in token_df_aggregate_group_pivot.columns)\
            and ("max_tokens" in token_df_aggregate_group_pivot.columns):
        gpu_waste_metrics_group_only = compute_GPU_waste_metrics(token_df_aggregate_group_pivot)

    gpu_utilization_metrics = spark.createDataFrame([], gpu_utilization_metrics_group_only.schema)
    gpu_waste_metrics = spark.createDataFrame([], gpu_utilization_metrics_group_only.schema)

    # if group_pivot_column_name is not null, then compute metrics for group_pivot as well
    if (group_pivot_column_name is not None):
        # compute GPU utilization metrics for group and group_pivot
        gpu_utilization_metrics = compute_GPU_utilization_metrics(token_df)
        # compute GPU waste metrics for group and group_pivot
        # These metrics are computed if we have max_tokens and finish_reason in the dataset
        if ("finish_reason" in token_df.columns) and ("max_tokens" in token_df.columns):
            gpu_waste_metrics = compute_GPU_waste_metrics(token_df)

    # Union the metrics
    GPU_token_stats_metrics = gpu_utilization_metrics.unionAll(gpu_waste_metrics)\
        .unionAll(gpu_utilization_metrics_group_only)\
        .unionAll(gpu_waste_metrics_group_only)

    # Add threshold_value column and set the value to null
    GPU_token_stats_metrics = GPU_token_stats_metrics.withColumn("threshold_value", lit(None).cast("float"))

    # Save metrics in default blob store and log it in active run
    save_spark_df_as_mltable(GPU_token_stats_metrics, args.signal_metrics)


if __name__ == "__main__":
    run()
