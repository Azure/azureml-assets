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
    arg_parser.add_argument("--signal_name", type=str)
    arg_parser.add_argument("--signal_type", type=str)
    arg_parser.add_argument("--signal_metrics", type=str)
    arg_parser.add_argument("--signal_output", type=str)

    args = arg_parser.parse_args()

    metrics: List[Row] = read_mltable_in_spark(args.signal_metrics).collect()

    metrics_dict = MetricOutputBuilder(metrics).get_metrics_dict()
    output_payload = self._to_metrics_output_payload(args.signal_name, args.signal_type, args.version, metrics_dict)

    local_path = str(uuid.uuid4())
    self._write_to_file(local_output_directory=local_path)

    target_remote_path = os.path.join(args.signal_output, "signals")
    amlfs_upload(local_path=local_path, remote_path=target_remote_path)

    print("*************** output metrics ***************")
    print("Successfully executed the output signal metric component.")

def _to_metrics_output_payload(self, signal_name: str, signal_type: str, version: str, metrics_dict: dict) -> dict:
    """Convert to a dictionary object for metric output."""
    output_payload = {
        "signalName": signal_name,
        "signalType": signal_type,
        "version": version,
        "metrics": metrics_dict,
    }
    return output_payload
    
def _write_to_file(self, local_output_directory: str):
    """Save the signal to a local directory."""
    os.makedirs(local_output_directory, exist_ok=True)
    signal_file = os.path.join(local_output_directory, f"{self.signal_name}.json")
    with open(signal_file, "w") as f:
        f.write(json.dumps(self.to_dict(), indent=4, default=np_encoder))

if __name__ == "__main__":
    run()
