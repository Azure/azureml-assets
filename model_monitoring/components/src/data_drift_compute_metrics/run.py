# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Metrics Spark Component."""

import argparse
from io_utils import (
    select_columns_from_spark_df,
    output_computed_measures_tests
)
from shared_utilities.io_utils import (
    read_mltable_in_spark
)
from compute_data_drift import (
    compute_data_drift_measures_tests
)

from shared_utilities.patch_mltable import patch_all
patch_all()


def run():
    """Compute metrics."""
    # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--signal_metrics', type=str)
    parser.add_argument('--production_dataset', type=str)
    parser.add_argument('--baseline_dataset', type=str)
    parser.add_argument('--categorical_metric', type=str)
    parser.add_argument('--numerical_metric', type=str)
    parser.add_argument('--categorical_threshold', type=str)
    parser.add_argument('--numerical_threshold', type=str)
    parser.add_argument('--feature_names', type=str, default=None)
    args = parser.parse_args()

    # Select columns
    select_columns = None
    if not (args.feature_names is not None):
        features = read_mltable_in_spark(args.feature_names)
        select_columns = [row['featureName'] for row in features.collect()]
        baseline_df = select_columns_from_spark_df(args.baseline_dataset, select_columns)
        production_df = select_columns_from_spark_df(args.production_dataset, select_columns)

    else:
        baseline_df = read_mltable_in_spark(args.baseline_dataset)
        production_df = read_mltable_in_spark(args.production_dataset)
        select_columns = baseline_df.columns

    metrics_df = compute_data_drift_measures_tests(baseline_df, production_df, args.numerical_metric, args.categorical_metric, args.numerical_threshold, args.categorical_threshold)
    # Save metrics in default blob store
    output_computed_measures_tests(metrics_df, args.signal_metrics)


if __name__ == "__main__":
    run()
