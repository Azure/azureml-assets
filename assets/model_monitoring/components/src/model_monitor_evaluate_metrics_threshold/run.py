# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Evaluate Metrics Threshold Component."""

import argparse
from evaluate_metrics_threshold import (
    evaluate_metrics_threshold,
)
from shared_utilities.io_utils import try_read_mltable_in_spark


def run():
    """Evaluate metrics threshold."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_to_evaluate", type=str)
    parser.add_argument("--signal_name", type=str)
    parser.add_argument("--notification_emails", type=str, required=False, nargs="?")
    args = parser.parse_args()

    metrics_df = try_read_mltable_in_spark(args.metrics_to_evaluate, "metrics_to_evaluate")

    if not metrics_df:
        print("No metrics to evaluate. Skipping metric evaluation.")
        return

    is_valid = evaluate_metrics_threshold(args.signal_name, metrics_df, args.notification_emails)

    if is_valid:
        print(
            "Successfully validated that the computed metrics are within the "
            + "given threshold for categorical and numerical metrics."
        )
    else:
        print("The metric evaluation job is completed.")


if __name__ == "__main__":
    run()
