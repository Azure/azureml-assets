# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Computing Metrics Spark Component."""

import argparse

from genai_token_statistics_compute_metrics.metrics_processor import MetricsProcessor
from shared_utilities.io_utils import (
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable
)


def run():
    """Execute the main function."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--production_dataset", type=str, required=True)
    parser.add_argument("--signal_metrics", type=str)
    parser.add_argument("--samples_index", type=str, required=True)
    args = parser.parse_args()

    token_df = try_read_mltable_in_spark_with_error(args.production_dataset, "production_dataset")

    processor = MetricsProcessor()
    processor.process(token_df)   
    metrics_df = processor.metrics_generator() 
    save_spark_df_as_mltable(metrics_df, args.signal_metrics)
    sample_date = processor.get_latest_run()
    save_spark_df_as_mltable(sample_date, args.samples_index)


if __name__ == "__main__":
    run()
