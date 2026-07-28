# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Prediction Drift signal output."""

from typing import List
from pyspark.sql import Row
from model_monitor_output_metrics.entities.feature_metrics import FeatureMetrics
from model_monitor_output_metrics.entities.signal_type import SignalType
from model_monitor_output_metrics.entities.signals.signal import Signal
from shared_utilities.run_metrics_utils import get_or_create_run_id, publish_metrics
from shared_utilities.df_utils import row_has_value, add_value_if_present


class DataQualitySignal(Signal):
    """Builder class which creates a Data Quality signal output."""

    def __init__(
        self,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
        feature_importance: List[Row]
    ):
        """Build Data Quality signal."""
        super().__init__(
            monitor_name, signal_name, "1.0.0", SignalType.DATA_QUALITY, metrics
        )
        self.row_count_metrics = None
        self.feature_metrics: List[FeatureMetrics] = self._build_feature_metrics(
            monitor_name, signal_name, metrics, feature_importance
        )

    def to_dict(self) -> dict:
        """Convert to a dictionary object."""
        signal_payload = {
            "signalName": self.signal_name,
            "signalType": self.signal_type.name,
            "version": self.version,
            "metrics": {"features": {}},
        }

        if self.row_count_metrics:
            signal_payload["metrics"]["rowCount"] = self.row_count_metrics

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
        if self.run_metrics is None or len(self.run_metrics) == 0:
            print("No run metric to publish.")
            return
        for run_metric in self.run_metrics:
            metrics = {
                "baseline_value": run_metric["baselineValue"],
                "target_value": run_metric["targetValue"],
                "threshold": run_metric["threshold"],
            }
            publish_metrics(run_id=run_metric["runId"], metrics=metrics, step=step)

    def _build_feature_metrics(
        self, monitor_name: str, signal_name: str, metrics: List[dict], feature_importance: List[Row]
    ) -> List[FeatureMetrics]:
        """Build feature-level metrics.

        Returns:
        List[FeatureMetrics]: The feature-level metrics.
        """
        if not metrics or len(metrics) == 0:
            return []

        featureimportance_dictionary = {}
        if feature_importance is not None:
            for row in feature_importance:
                featureimportance_dictionary[row[0]] = row[1]

        output = {}

        for metric in metrics:
            if metric["metric_name"] == "RowCount" and metric["feature_name"] == "":
                self.row_count_metrics = {
                    "baselineCount": metric["baseline_metric_value"],
                    "targetCount": metric["target_metric_value"],
                }

            if not row_has_value(metric, "feature_name"):
                continue

            feature_name = metric["feature_name"]
            if feature_name not in output:
                output[feature_name] = FeatureMetrics(
                    metric["data_type"],
                    feature_name,
                    metrics=[],
                    run_metrics=[],
                    histogram=None,
                )

            # Add the run metrics
            run_id = get_or_create_run_id(
                monitor_name, signal_name, feature_name, metric["metric_name"]
            )
            run_metric = {
                "runId": run_id,
                "targetValue": None if metric["target_metric_value"] is None
                else float(metric["target_metric_value"]),
                "baselineValue": None if metric["baseline_metric_value"] is None
                else float(metric["baseline_metric_value"]),
            }
            run_metric = add_value_if_present(
                metric, "threshold_value", run_metric, "threshold"
            )
            self.run_metrics.append(run_metric)
            output[feature_name].run_metrics.append(
                {
                    "runId": run_id,
                    "metricName": metric["metric_name"],
                    "runMetricName": "value",
                }
            )

            # Add the feature metrics
            feature_metrics = {
                "metricName": metric["metric_name"],
                "targetValue": metric["target_metric_value"],
                "baselineValue": metric["baseline_metric_value"],
            }
            feature_metrics = add_value_if_present(
                metric, "threshold_value", feature_metrics, "threshold"
            )
            output[feature_name].metrics.append(feature_metrics)

            # Add the feature importance metrics
            if len(featureimportance_dictionary) != 0 and feature_name in featureimportance_dictionary:
                if not any('BaselineFeatureImportance' in d.values() for d in output[feature_name].metrics):
                    feature_importance_metric = {
                        "metricName": "BaselineFeatureImportance",
                        "metricValue": featureimportance_dictionary[feature_name],
                    }
                    output[feature_name].metrics.append(feature_importance_metric)

        return list(output.values())
