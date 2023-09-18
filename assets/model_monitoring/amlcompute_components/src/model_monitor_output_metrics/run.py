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
from shared_utilities.patch_mltable import patch_all

patch_all()

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
    target_remote_path = _convert_to_azureml_uri(target_remote_path)
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)

    print("*************** output metrics ***************")
    print("Successfully executed the output metric component.")


def _convert_to_azureml_uri(local_uri: str):
    import os
    import re
    print(os.environ)
    workspace_scope_var = os.environ['AZUREML_WORKSPACE_SCOPE']
    workspace_scope = workspace_scope_var.replace("/subscriptions", "subscriptions").replace("/providers/Microsoft.MachineLearningServices", "").lower()
    datastore_id = os.path.join("azureml://", workspace_scope, "datastores/workspaceblobstore")
    job_id = os.environ['AZUREML_RUN_ID']
    pattern = r"^(.+?)/cap/data-capability/wd/(.+?)$"
    match = re.search(pattern, local_uri)
    output_folder = match.group(2)
    azureml_uri = os.path.join(datastore_id, "paths/azureml", job_id, output_folder)
    print(f"_DataframeWriterManager.write: workspace_scope_var: {workspace_scope_var}, workspace_scope: {workspace_scope}, local_uri: {local_uri}, output_folder: {output_folder}, datastore_id: {datastore_id}, azureml_uri: {azureml_uri}")
    
    return azureml_uri


if __name__ == "__main__":
    run()
