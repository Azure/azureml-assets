# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Generation Safety and Quality signal output."""

from typing import List
import json
import os
from pyspark.sql import Row
from model_monitor_output_metrics.entities.row_count_metrics import RowCountMetrics
from model_monitor_output_metrics.entities.signal_type import SignalType
from model_monitor_output_metrics.entities.signals.signal import Signal
from shared_utilities.run_metrics_utils import get_or_create_run_id, publish_metric
from shared_utilities.io_utils import np_encoder


class GenerationSafetyQualitySignal(Signal):
    """Builder class which creates a Feature Attribution signal output."""

    def __init__(
        self,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
    ):
        """Build Feature Attribution Drift signal."""
        super().__init__(
            monitor_name,
            signal_name,
            "1.0.0",
            SignalType.FEATURE_ATTRIBUTION_DRIFT,
            metrics,
        )
        self.row_count_metrics = RowCountMetrics(metrics)
        self._build_metrics(monitor_name, signal_name, metrics)

    def to_dict(self) -> dict:
        """Convert to a dictionary object."""
        signal_payload = {
            "signalName": self.signal_name,
            "signalType": self.signal_type.name,
            "version": self.version,
            "metrics": self.global_metrics,
        }

        if self.row_count_metrics.has_value():
            signal_payload["metrics"]["rowCount"] = self.row_count_metrics.to_dict()

        signal_payload["metrics"]["features"] = {}

        return signal_payload

    def to_file(self, local_output_directory: str):
        """Save the signal to a local directory."""
        super().to_file(local_output_directory)
        # Output histograms to file
        histogram_directory = os.path.join(local_output_directory, self.signal_name)
        os.makedirs(histogram_directory, exist_ok=True)

        histogram_file = os.path.join(
            histogram_directory, "AcceptableGroundednessScorePerInstance.histogram.json"
        )
        with open(histogram_file, "w") as f:
            f.write(
                json.dumps(
                    self.histogram,
                    indent=4,
                    default=np_encoder,
                )
            )

    def publish_metrics(self, step: int):
        """Publish metrics to AML Run Metrics."""
        run_metric = self.global_metrics["AggregatedGroundednessPassRate"]
        publish_metric(
            run_metric["runId"],
            float(run_metric["metricValue"]),
            float(run_metric["threshold"]),
            step,
        )

    def _build_metrics(self, monitor_name: str, signal_name: str, metrics: List[dict]):
        """Build metrics."""
        rows = []
        global_metric_cache = {
            "AcceptableGroundednessScorePerInstance": {
                "threshold": 0,
                "histogram": f"signals/{self.signal_name}/AcceptableGroundednessScorePerInstance.histogram.json",
            }
        }
        self.histogram = {
            "histogram": [],
            "featureName": "AcceptableGroundednessScorePerInstance"
        }
        for metric in metrics:
            if "GroundednessCount_" in metric["metric_name"]:
                bucket = metric["metric_name"].split("_")[1]
                rows.append(
                    Row(
                        feature_bucket="Groundedness",
                        bucket_count=metric["metric_value"],
                        data_type="categorical",
                        category_bucket=bucket,
                    )
                )
                histogram_bucket = {
                    "category": bucket,
                    "baselineCount": metric["metric_value"],
                }
                self.histogram["histogram"].append(histogram_bucket)

            if metric["metric_name"] == "AggregatedGroundednessPassRate":
                global_metric_cache[metric["metric_name"]] = {
                    "metricValue": metric["metric_value"],
                    "threshold": 0,
                    "runId": get_or_create_run_id(
                        monitor_name=monitor_name,
                        signal_name=signal_name,
                        feature_name=None,
                        metric_name="AggregatedGroundednessPassRate",
                    ),
                    "metricName": metric["metric_name"],
                    "runMetricName": "value",
                }

        self.global_metrics = global_metric_cache
