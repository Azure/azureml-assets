# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Output Metrics Component."""

import argparse
import os
from typing import List
import uuid
from pyspark.sql import Row
from dateutil import parser
from shared_utilities.io_utils import read_mltable_in_spark
from model_monitor_output_metrics.factories.signal_factory import SignalFactory
from model_monitor_output_metrics.entities.signals.signal import Signal
from shared_utilities.amlfs import amlfs_upload


def run():
    """Output metrics."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--monitor_name", type=str)
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--signal_type", type=str)
    arg_parser.add_argument("--metric_timestamp", type=str)
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--baseline_histogram", type=str, required=False)
    arg_parser.add_argument("--target_histogram", type=str, required=False)
    arg_parser.add_argument("--signal_output", type=str)

    args = arg_parser.parse_args()

    metrics: List[Row] = read_mltable_in_spark(args.signal_metrics).collect()

    baseline_histogram = None
    target_histogram = None

    if args.baseline_histogram is not None:
        baseline_histogram: List[Row] = read_mltable_in_spark(
            args.baseline_histogram
        ).collect()

    if args.target_histogram is not None:
        target_histogram: List[Row] = read_mltable_in_spark(
            args.target_histogram
        ).collect()

    signal: Signal = SignalFactory().produce(
        signal_type=args.signal_type,
        monitor_name=args.monitor_name,
        signal_name=args.signal_name,
        metrics=metrics,
        baseline_histogram=baseline_histogram,
        target_histogram=target_histogram,
    )

    metric_timestamp = parser.parse(args.metric_timestamp)
    signal.publish_metrics(step=int(metric_timestamp.timestamp()))

    local_path = str(uuid.uuid4())
    signal.to_file(local_output_directory=local_path)

    target_remote_path = os.path.join(args.signal_output, "signals")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)

    print("*************** output metrics ***************")
    print("Successfully executed the output metric component.")


if __name__ == "__main__":
    run()
