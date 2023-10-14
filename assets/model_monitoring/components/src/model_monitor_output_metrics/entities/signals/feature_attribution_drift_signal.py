# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Feature Attribution signal output."""

from typing import List
from pyspark.sql import Row
from model_monitor_output_metrics.entities.feature_metrics import FeatureMetrics
from model_monitor_output_metrics.entities.row_count_metrics import RowCountMetrics
from model_monitor_output_metrics.entities.signal_type import SignalType
from model_monitor_output_metrics.entities.signals.signal import Signal
from shared_utilities.run_metrics_utils import get_or_create_run_id
from shared_utilities.df_utils import row_has_value, add_value_if_present


class FeatureAttributionDriftSignal(Signal):
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
        self.global_metrics = {}
        self.feature_metrics = []
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
        signal_payload["metrics"]["features"] = {}

        if self.row_count_metrics.has_value():
            signal_payload["metrics"]["rowCount"] = self.row_count_metrics.to_dict()

        for feature_metric in self.feature_metrics:
            signal_payload["metrics"]["features"][
                feature_metric.feature_name
            ] = feature_metric.to_dict()
        return signal_payload

    def to_file(self, local_output_directory: str):
        """Save the signal to a local directory."""
        super().to_file(local_output_directory)

    def publish_metrics(self, step: int):
        """Publish metrics to AML Run Metrics."""
        super().publish_metrics(step=step)

    def _build_metrics(self, monitor_name: str, signal_name: str, metrics: List[dict]):
        """Build metrics."""
        if not metrics or len(metrics) == 0:
            return []

        feature_metric_cache = {}
        global_metric_cache = {}
        for metric in metrics:
            if metric["metric_name"] == "NormalizedDiscountedCumulativeGain":

                run_id = get_or_create_run_id(
                    monitor_name=monitor_name,
                    signal_name=signal_name,
                    feature_name=None,
                    metric_name=metric["metric_name"],
                )
                run_metric = {
                    "runId": run_id,
                    "value": metric["metric_value"],
                }
                run_metric = add_value_if_present(
                    metric, "threshold_value", run_metric, "threshold"
                )
                self.run_metrics.append(run_metric)

                global_metric = {
                    "metricValue": metric["metric_value"],
                    "runId": run_id,
                    "metricName": metric["metric_name"],
                    "runMetricName": "value",
                }
                global_metric = add_value_if_present(
                    metric, "threshold_value", global_metric, "threshold"
                )
                global_metric_cache[metric["metric_name"]] = global_metric

            if not row_has_value(metric, "feature_name"):
                continue

            feature_name = metric["feature_name"]
            if feature_name not in feature_metric_cache:
                feature_metric_cache[feature_name] = FeatureMetrics(
                    metric["data_type"],
                    feature_name,
                    metrics=[],
                    run_metrics=[],
                    histogram=None,
                )
            feature_metric_cache[feature_name].metrics.append(
                {
                    "metricName": metric["metric_name"],
                    "metricValue": metric["metric_value"],
                }
            )

        self.feature_metrics = list(feature_metric_cache.values())
        self.global_metrics = global_metric_cache
