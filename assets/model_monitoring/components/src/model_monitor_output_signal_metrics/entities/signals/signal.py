# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Data Drift signal output."""
import os
import json
from typing import List
from pyspark.sql import Row
from shared_utilities.io_utils import np_encoder
from model_monitor_output_signal_metrics.entities.signal_type import SignalType
from model_monitor_output_signal_metrics.builder.signals.metric_output_builder import MetricOutputBuilder
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
        self.metrics_dict = MetricOutputBuilder(metrics)

    def to_dict(self) -> dict:
        """Convert to a dictionary object."""
        signal_payload = {
            "signalName": self.signal_name,
            "signalType": self.signal_type.name,
            "version": self.version,
            "metrics": self.metrics_dict,
        }
        return signal_payload

    def to_file(self, local_output_directory: str):
        """Save the signal to a local directory."""
        os.makedirs(local_output_directory, exist_ok=True)
        signal_file = os.path.join(local_output_directory, f"{self.signal_name}.json")
        with open(signal_file, "w") as f:
            f.write(json.dumps(self.to_dict(), indent=4, default=np_encoder))
