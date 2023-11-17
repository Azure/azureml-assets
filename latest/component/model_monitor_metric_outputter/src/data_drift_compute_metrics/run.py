# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
from io_utils import (
    select_columns_from_spark_df,
    output_computed_measures_tests,
)
from shared_utilities.io_utils import try_read_mltable_in_spark_with_error
from compute_data_drift import compute_data_drift_measures_tests


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_metrics", type=str)
    parser.add_argument("--production_dataset", type=str)
    parser.add_argument("--baseline_dataset", type=str)
    parser.add_argument("--categorical_metric", type=str)
    parser.add_argument("--numerical_metric", type=str)
    parser.add_argument("--categorical_threshold", type=str)
    parser.add_argument("--numerical_threshold", type=str)
    parser.add_argument("--feature_names", type=str, default=None)
    args = parser.parse_args()

    # Select columns
    select_columns = None

    baseline_df = try_read_mltable_in_spark_with_error(
        args.baseline_dataset, "baseline_dataset"
    )
    production_df = try_read_mltable_in_spark_with_error(
        args.production_dataset, "production_dataset"
    )

    if args.feature_names:
        features = try_read_mltable_in_spark_with_error(args.feature_names, "feature_names")

        select_columns = [row["featureName"] for row in features.collect()]
        baseline_df = select_columns_from_spark_df(baseline_df, select_columns)
        production_df = select_columns_from_spark_df(production_df, select_columns)

    metrics_df = compute_data_drift_measures_tests(
        baseline_df,
        production_df,
        args.numerical_metric,
        args.categorical_metric,
        args.numerical_threshold,
        args.categorical_threshold,
    )
    # Save metrics in default blob store
    output_computed_measures_tests(metrics_df, args.signal_metrics)


if __name__ == "__main__":
    run()
