# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Evaluate Metrics Threshold Component."""

import argparse
from evaluate_metrics_threshold import (
    evaluate_metrics_threshold,
)
from shared_utilities.io_utils import read_mltable_in_spark


def evaluate_metrics(
    signal_name: str, metrics_to_evaluate: str, notification_emails: str
):
    """Evaluate the computed metrics against the threshold."""
    metrics_to_evaluate_df = read_mltable_in_spark(metrics_to_evaluate)

    evaluate_metrics_threshold(signal_name, metrics_to_evaluate_df, notification_emails)


def run():
    """Evaluate metrics threshold."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_to_evaluate", type=str)
    parser.add_argument("--signal_name", type=str)
    parser.add_argument("--notification_emails", type=str, required=False, nargs="?")
    args = parser.parse_args()

    is_valid = evaluate_metrics(
        args.signal_name, args.metrics_to_evaluate, args.notification_emails
    )

    if is_valid:
        print(
            "Successfully validated that the computed metrics are within the given threshold for categorical and numerical metrics."
        )
    else:
        print("The metric evaluation job is completed.")


if __name__ == "__main__":
    run()
