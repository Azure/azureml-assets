# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Computing Metrics Spark Component."""

import argparse
import os

from pyspark.sql import functions as F
from pyspark.sql.types import StructField, StructType, StringType
from genai_token_statistics_compute_metrics.metrics_processor import MetricsProcessor
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark
)
from shared_utilities.constants import (
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE
)
from genai_token_statistics_compute_metrics.constants import (
    TOTAL_TOKEN_COUNT,
    TOTAL_PROMPT_COUNT,
    TOTAL_COMPLETION_COUNT,
    AVG_TOKEN_COUNT,
    AVG_PROMPT_COUNT,
    AVG_COMPLETION_COUNT)


def run():
    """Execute the main function."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--signal_metrics", type=str)
    parser.add_argument("--token_count", type=str, required=True)
    parser.add_argument("--samples_index", type=str, required=True)
    args = parser.parse_args()

    token_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    processor = MetricsProcessor()
    processor.process(token_df)
    metrics_df = processor.metrics_df

    flat_metrics_df = metrics_df.select(
        F.array(
            F.struct(F.lit(TOTAL_TOKEN_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('total_token_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)),
            F.struct(F.lit(TOTAL_PROMPT_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('total_prompt_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)),
            F.struct(F.lit(TOTAL_COMPLETION_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('total_completion_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)),
            F.struct(F.lit(AVG_TOKEN_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('avg_total_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)),
            F.struct(F.lit(AVG_PROMPT_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('avg_prompt_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)),
            F.struct(F.lit(AVG_COMPLETION_COUNT).alias(SIGNAL_METRICS_METRIC_NAME),
                     F.col('avg_completion_count').alias(SIGNAL_METRICS_METRIC_VALUE),
                     F.lit(0.0).alias(SIGNAL_METRICS_THRESHOLD_VALUE)))
        .alias('metrics')).select(F.explode(F.col('metrics'))
                                  .alias('metrics')).select(F.col('metrics')[SIGNAL_METRICS_METRIC_NAME]
                                                            .alias(SIGNAL_METRICS_METRIC_NAME),
                                                            F.col('metrics')[SIGNAL_METRICS_METRIC_VALUE]
                                                            .alias(SIGNAL_METRICS_METRIC_VALUE),
                                                            F.col('metrics')[SIGNAL_METRICS_THRESHOLD_VALUE]
                                                            .alias(SIGNAL_METRICS_THRESHOLD_VALUE)
                                                            )
    flat_metrics_df.show()
    flat_metrics_rdd = processor.metrics_model_df.rdd.flatMap(lambda row: [
        (f"{row['model']}.CompletionCount", row['total_completion_count'], 0),
        (f"{row['model']}.TotalRequests", row['total_requests'], 0),
        (f"{row['model']}.PromptCount", row['total_prompt_count'], 0)
    ])
    flat_model_metrics_df = flat_metrics_rdd.toDF([SIGNAL_METRICS_METRIC_NAME, SIGNAL_METRICS_METRIC_VALUE,
                                                   SIGNAL_METRICS_THRESHOLD_VALUE])

    flat_metrics_df = flat_model_metrics_df.union(flat_metrics_df)
    save_spark_df_as_mltable(flat_metrics_df, args.signal_metrics)
    sample_data = processor.requests_df
    save_spark_df_as_mltable(sample_data, args.token_count)
    samples_index_rows = []
    samples_index_rows.append({"metric_name": "TokenCount",
                               "group": "",
                               "group_dimension": "",
                               "samples_name": "TokenCount",
                               "asset": f"azureml_{os.environ['AZUREML_RUN_ID']}_output_data_token_count:1"})
    spark = init_spark()
    metadata_schema = StructType(
        [
            StructField("metric_name", StringType(), True),
            StructField("group", StringType(), True),
            StructField("group_dimension", StringType(), True),
            StructField("samples_name", StringType(), True),
            StructField("asset", StringType(), True),
        ]
    )
    sample_df = spark.createDataFrame(samples_index_rows, metadata_schema)
    save_spark_df_as_mltable(sample_df, args.samples_index)


if __name__ == "__main__":
    run()
