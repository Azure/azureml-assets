# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Annotation Metrics Computing Spark Component."""

import argparse
import re
from pyspark.sql.functions import col, lit, sum, udf
from pyspark.sql.types import IntegerType, StructField, StructType, StringType
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
    try_read_mltable_in_spark_with_warning,
)

GROUP_COLUMN = "group"
GROUP_DIMENSION_COLUMN = "group_dimension"
METRIC_NAME_COLUMN = "metric_name"
METRIC_VALUE_COLUMN = "metric_value"
THRESHOLD_COLUMN = "threshold_value"

THRESHOLD_PARAMS = [
    "groundedness_passrate_threshold",
    "similarity_passrate_threshold",
    "relevance_passrate_threshold",
    "fluency_passrate_threshold",
    "coherence_passrate_threshold",
]

ALL_METRIC_NAMES = [
    "AcceptableGroundednessScorePerInstance",
    "AggregatedGroundednessPassRate",
    "AcceptableCoherenceScorePerInstance",
    "AggregatedCoherencePassRate",
    "AcceptableFluencyScorePerInstance",
    "AggregatedFluencyPassRate",
    "AcceptableSimilarityScorePerInstance",
    "AggregatedSimilarityPassRate",
    "AcceptableRelevanceScorePerInstance",
    "AggregatedRelevancePassRate",
]


def _get_per_instance_threshold(df, metric_name):
    return df.filter(col(METRIC_NAME_COLUMN).contains(metric_name)).select(THRESHOLD_COLUMN
                                                                           ).collect()[0][THRESHOLD_COLUMN]


def _calculate_passrate(df, metric_name):
    threshold = _get_per_instance_threshold(df, metric_name)

    df_with_buckets = df.filter(
        col(METRIC_NAME_COLUMN).contains(metric_name)
    ).withColumn(
        "bucket",
        udf(lambda group: int(group), IntegerType())(
            col(GROUP_COLUMN)
        ),
    )
    passing = (
        df_with_buckets.filter(col("bucket") >= int(threshold))
        .select(sum(METRIC_VALUE_COLUMN))
        .head()[0]
    )
    total = df_with_buckets.select(sum(METRIC_VALUE_COLUMN)).head()[0]
    if total == 0:
        return "1"
    passrate = passing / total
    return str(passrate)


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric_names", type=str)
    parser.add_argument("--annotation_histogram", type=str)
    parser.add_argument("--signal_metrics", type=str)

    parser.add_argument("--groundedness_passrate_threshold", type=float, default=0.7)
    parser.add_argument("--similarity_passrate_threshold", type=float, default=0.7)
    parser.add_argument("--relevance_passrate_threshold", type=float, default=0.7)
    parser.add_argument("--fluency_passrate_threshold", type=float, default=0.7)
    parser.add_argument("--coherence_passrate_threshold", type=float, default=0.7)

    args = parser.parse_args()
    histogram_df = try_read_mltable_in_spark_with_warning(args.annotation_histogram, "annotation_histogram")

    if not histogram_df:
        print("No histogram to annotate. Skipping computing annotation metrics.")
        return

    spark = init_spark()
    # Cast to float because metric_value was integer so far
    # but we're adding percentages now.
    histogram_df = histogram_df.withColumn(
        METRIC_VALUE_COLUMN, histogram_df[METRIC_VALUE_COLUMN].cast("float")
    )
    threshold_args = {
        arg: getattr(args, arg) for arg in THRESHOLD_PARAMS if hasattr(args, arg)
    }
    # remove all but groundedness/fluency/coherence/relevance/similarity from metric names and
    # remove duplicates
    input_metric_names = [m.strip() for m in args.metric_names.split(",")]
    pruned_metric_names = [re.sub(r'^(.*?)(Groundedness|Fluency|Coherence|Relevance|Similarity)(.*?)$', r'\2', m) for
                           m in input_metric_names]
    compact_metric_names = list(set(pruned_metric_names))

    aggregated_metrics_df = histogram_df.withColumn(GROUP_DIMENSION_COLUMN, lit(""))

    # overwrite threshold value column
    aggregated_metrics_df = aggregated_metrics_df.withColumn(THRESHOLD_COLUMN, lit("nan").cast("float"))
    metadata_schema = StructType(
            [
                StructField(GROUP_COLUMN, StringType(), True),
                StructField(METRIC_VALUE_COLUMN, StringType(), True),
                StructField(METRIC_NAME_COLUMN, StringType(), True),
                StructField(THRESHOLD_COLUMN, StringType(), True),
                StructField(GROUP_DIMENSION_COLUMN, StringType(), True)
            ]
        )
    for metric_name in compact_metric_names:
        passrate_threshold = threshold_args[f"{metric_name.lower()}_passrate_threshold"]
        print(passrate_threshold)
        print("line 126")
        full_pass_rate_metric_name = f"Aggregated{metric_name}PassRate"
        full_per_instance_score_metric_name = f"Acceptable{metric_name}ScorePerInstance"
        if full_pass_rate_metric_name in input_metric_names:
            metric_df = spark.createDataFrame(
                    [
                        (
                            "",
                            _calculate_passrate(histogram_df, metric_name),
                            full_pass_rate_metric_name,
                            passrate_threshold,
                            ""
                        )
                    ],
                    metadata_schema,
                )
            aggregated_metrics_df = aggregated_metrics_df.union(metric_df)
        if full_per_instance_score_metric_name not in input_metric_names:
            aggregated_metrics_df = aggregated_metrics_df.filter(col(METRIC_NAME_COLUMN)
                                                                 != full_per_instance_score_metric_name)
        else:
            threshold = _get_per_instance_threshold(histogram_df, metric_name)
            threshold_row = spark.createDataFrame(
                    [
                        (
                            "",
                            "",
                            full_per_instance_score_metric_name,
                            threshold,
                            "",
                        )
                    ],
                    metadata_schema,
                )
            aggregated_metrics_df = aggregated_metrics_df.union(threshold_row)
    save_spark_df_as_mltable(aggregated_metrics_df, args.signal_metrics)


if __name__ == "__main__":
    run()
