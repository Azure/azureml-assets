# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Monitor Metric Outputter Component."""

import argparse
from dateutil import parser
import json
import os
import uuid

from pyspark.sql import Row
from typing import List

from model_monitor_metric_outputter.builder.metric_output_builder import (
    MetricOutputBuilder,
)
from model_monitor_metric_outputter.builder.samples_output_builder import (
    SamplesOutputBuilder,
)
from model_monitor_metric_outputter.runmetric_client import RunMetricClient
from model_monitor_metric_outputter.runmetrics_publisher import RunMetricPublisher
from shared_utilities.amlfs import amlfs_upload
from shared_utilities.constants import METADATA_VERSION
from shared_utilities.dict_utils import merge_dicts
from shared_utilities.io_utils import (
    np_encoder,
    try_read_mltable_in_spark_with_error,
)


def run():
    """Output metrics."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--monitor_name", type=str)
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--signal_type", type=str)
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--samples_index", type=str, required=False, nargs="?")
    arg_parser.add_argument("--metric_timestamp", type=str)
    arg_parser.add_argument("--signal_output", type=str)

    args = arg_parser.parse_args()
    args_dict = vars(args)

    signal_metrics = try_read_mltable_in_spark_with_error(args.signal_metrics, "signal_metrics")

    metrics: List[Row] = signal_metrics.collect()

    runmetric_client = RunMetricClient()
    result = MetricOutputBuilder(
        runmetric_client, args.monitor_name, args.signal_name, metrics
    ).get_metrics_dict()

    samples_index: List[Row] = None
    samples_index_df = None
    if args_dict["samples_index"]:
        try:
            print("Processing samples index.")
            samples_index_df = try_read_mltable_in_spark_with_error(
                args.samples_index, "samples_index"
            )
            samples_index: List[Row] = samples_index_df.collect()
            result = merge_dicts(
                result, SamplesOutputBuilder(samples_index).get_samples_dict()
            )
        except Exception as e:
            # whatever error we got, including DataNotFoundError, skip processing the samples index
            print(f"Samples index is empty. Skipping processing of the samples index. {e}")

    output_payload = to_output_payload(args.signal_name, args.signal_type, result)

    local_path = str(uuid.uuid4())
    write_to_file(
        payload=output_payload,
        local_output_directory=local_path,
        signal_name=args.signal_name,
    )

    target_remote_path = os.path.join(args.signal_output, "signals")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)

    print("Uploading run metrics to AML run history.")
    metric_timestamp = parser.parse(args.metric_timestamp)
    metric_step = int(metric_timestamp.timestamp())
    print(f"Publishing metrics with step '{metric_step}'.")

    RunMetricPublisher(runmetric_client).publish_metrics(result, metric_step)

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
