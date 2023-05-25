# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Quality Metrics Joiner Spark Component."""

import argparse
from pyspark.sql.functions import col, when
from shared_utilities.io_utils import read_mltable_in_spark, save_spark_df_as_mltable


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_metrics", type=str)
    parser.add_argument("--target_metrics", type=str)
    parser.add_argument("--signal_metrics", type=str)
    parser.add_argument("--numerical_threshold", type=str)
    parser.add_argument("--categorical_threshold", type=str)
    args = parser.parse_args()

    print("args.numerical_threshold: ", args.numerical_threshold)
    print("args.categorical_threshold: ", args.categorical_threshold)
    baseline_df = (
        read_mltable_in_spark(args.baseline_metrics)
        .withColumn("baseline_metric_value", col("metricValue"))
        .drop("violationCount")
        .drop("metricValue")
    )
    target_df = (
        read_mltable_in_spark(args.target_metrics)
        .withColumn("target_metric_value", col("metricValue"))
        .drop("dataType")
        .drop("violationCount")
        .drop("metricValue")
    )
    result_df = baseline_df.join(
        target_df, on=["featureName", "metricName"], how="inner"
    )

    result_df = (
        result_df.withColumnRenamed("featureName", "feature_name")
        .withColumnRenamed("metricName", "metric_name")
        .withColumnRenamed("dataType", "data_type")
        .withColumn("metric_value", col("target_metric_value"))
    )

    result_df = result_df.withColumn(
        "threshold_value",
        when(
            result_df.data_type == "Numerical",
            col("baseline_metric_value") + args.numerical_threshold,
        ).otherwise(col("baseline_metric_value") + args.categorical_threshold),
    )

    result_df.show()
    save_spark_df_as_mltable(result_df, args.signal_metrics)


if __name__ == "__main__":
    run()
