# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Data Drift signal output."""
import os
import json
from typing import List
from pyspark.sql import Row
from shared_utilities.io_utils import np_encoder
from model_monitor_output_metrics.entities.signal_type import SignalType
from shared_utilities.run_metrics_utils import publish_metric


class Signal:
    """Builder class which creates a signal output."""

    def __init__(
        self,
        monitor_name: str,
        signal_name: str,
        version: str,
        signal_type: SignalType,
        metrics: List[Row],
    ):
        """Build DataDrift signal."""
        self.monitor_name: str = monitor_name
        self.signal_name: str = signal_name
        self.version: str = version
        self.signal_type: SignalType = signal_type
        self.metrics: List[Row] = metrics
        self.run_metrics = []

    def to_file(self, local_output_directory: str):
        """Save the signal to a local directory."""
        os.makedirs(local_output_directory, exist_ok=True)
        signal_file = os.path.join(local_output_directory, f"{self.signal_name}.json")
        with open(signal_file, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=4, default=np_encoder))

    def publish_metrics(self, step: int):
        """Publish metrics to AML Run Metrics."""
        if self.run_metrics is None or len(self.run_metrics) == 0:
            print("No run metric to publish.")
            return
        for run_metric in self.run_metrics:
            threshold = None
            if "threshold" in run_metric:
                threshold = run_metric["threshold"]
            if "value" in run_metric and run_metric["value"]:
                publish_metric(
                run_metric["runId"], float(run_metric["value"]), threshold, step
                )

    def to_dict(self) -> dict:
        """Convert to a dictionary object."""
        pass
