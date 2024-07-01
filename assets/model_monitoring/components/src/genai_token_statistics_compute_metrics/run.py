# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Computing Metrics Spark Component."""

import argparse
<<<<<<< HEAD

from genai_token_statistics_compute_metrics.metrics_processor import MetricsProcessor
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable
=======
import os

from pyspark.sql.types import StructField, StructType, StringType
from genai_token_statistics_compute_metrics.metrics_processor import MetricsProcessor
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
    init_spark
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
)


def run():
    """Execute the main function."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--signal_metrics", type=str)
<<<<<<< HEAD
=======
    parser.add_argument("--token_count", type=str, required=True)
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
    parser.add_argument("--samples_index", type=str, required=True)
    args = parser.parse_args()

    token_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    processor = MetricsProcessor()
    processor.process(token_df)
    metrics_df = processor.metrics_generator()
    save_spark_df_as_mltable(metrics_df, args.signal_metrics)
<<<<<<< HEAD
    sample_date = processor.get_latest_run()
    save_spark_df_as_mltable(sample_date, args.samples_index)
=======
    sample_data = processor.get_latest_run()
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
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa


if __name__ == "__main__":
    run()
