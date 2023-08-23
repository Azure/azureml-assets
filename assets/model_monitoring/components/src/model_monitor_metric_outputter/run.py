# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Metric Outputter Component."""

import argparse
import json
import os
import uuid

from pyspark.sql import Row
from typing import List

from model_monitor_metric_outputter.builder.metric_output_builder import MetricOutputBuilder
from shared_utilities.amlfs import amlfs_upload
from shared_utilities.constants import METADATA_VERSION
from shared_utilities.io_utils import (
    np_encoder,
    read_mltable_in_spark,
)


def run():
    """Output metrics."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--signal_type", type=str)
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--signal_output", type=str)

    args = arg_parser.parse_args()

    metrics: List[Row] = read_mltable_in_spark(args.signal_metrics).collect()

    metrics_dict = MetricOutputBuilder(metrics).get_metrics_dict()
    output_payload = to_output_payload(args.signal_name, args.signal_type, metrics_dict)

    local_path = str(uuid.uuid4())
    write_to_file(payload=output_payload, local_output_directory=local_path, signal_name=args.signal_name)

    target_remote_path = os.path.join(args.signal_output, "signals")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)

    print("*************** output metrics ***************")
    print("Successfully executed the metric outputter component.")


def to_output_payload(signal_name: str, signal_type: str, metrics_dict: dict) -> dict:
    """Convert to a dictionary object for metric output."""
    output_payload = {
        "signalName": signal_name,
        "signalType": signal_type,
        "version": METADATA_VERSION,
        "metrics": metrics_dict,
    }
    return output_payload


def write_to_file(payload: dict, local_output_directory: str, signal_name: str):
    """Save the signal to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    signal_file = os.path.join(local_output_directory, f"{signal_name}.json")
    with open(signal_file, "w") as f:
        f.write(json.dumps(payload, indent=4, default=np_encoder))


if __name__ == "__main__":
    run()
