# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Builder class which creates a Data Drift signal output."""

import json
import os
from typing import List
from pyspark.sql import Row
from model_monitor_output_metrics.entities.feature_metrics import FeatureMetrics
from model_monitor_output_metrics.entities.row_count_metrics import RowCountMetrics
from model_monitor_output_metrics.entities.signal_type import SignalType
from model_monitor_output_metrics.entities.signals.signal import Signal
from model_monitor_output_metrics.builders.histogram_builder import HistogramBuilder
from shared_utilities.io_utils import np_encoder
from shared_utilities.run_metrics_utils import get_or_create_run_id
from shared_utilities.df_utils import row_has_value, add_value_if_present


class DataDriftSignal(Signal):
    """Builder class which creates a Data Drift signal output."""

    def __init__(
        self,
        monitor_name: str,
        signal_name: str,
        metrics: List[Row],
        baseline_histogram: List[Row],
        target_histogram: List[Row],
        feature_importance: List[Row]
    ):
        """Build DataDrift signal."""
        super().__init__(
            monitor_name, signal_name, "1.0.0", SignalType.DATA_DRIFT, metrics
        )
        self.row_count_metrics = RowCountMetrics(metrics)
        self.feature_metrics: List[FeatureMetrics] = self._build_feature_metrics(
            monitor_name, signal_name, metrics, feature_importance
        )
        self.histogram_builder = HistogramBuilder(baseline_histogram, target_histogram)

    def to_dict(self) -> dict:
        """Convert to a dictionary object."""
        signal_payload = {
            "signalName": self.signal_name,
            "signalType": self.signal_type.name,
            "version": self.version,
            "metrics": {"features": {}},
        }

        if self.row_count_metrics.has_value():
            signal_payload["metrics"]["rowCount"] = self.row_count_metrics.to_dict()

        for feature_metric in self.feature_metrics:
            signal_payload["metrics"]["features"][
                feature_metric.feature_name
            ] = feature_metric.to_dict()

        return signal_payload

    def publish_metrics(self, step: int):
        """Publish metrics to AML Run Metrics."""
        super().publish_metrics(step=step)

    def to_file(self, local_output_directory: str):
        """Save the signal to a local directory."""
        super().to_file(local_output_directory)

        # Output histograms to file
        histogram_directory = os.path.join(local_output_directory, self.signal_name)
        os.makedirs(histogram_directory, exist_ok=True)
        for feature in self.histogram_builder.get_features():
            histogram_file = os.path.join(
                histogram_directory, f"{feature}.histogram.json"
            )
            with open(histogram_file, "w") as f:
                f.write(
                    json.dumps(
                        self.histogram_builder.build(feature),
                        indent=4,
                        default=np_encoder,
                    )
                )

    def _build_feature_metrics(
        self, monitor_name: str, signal_name: str, metrics: List[dict], feature_importance: List[Row]
    ) -> List[FeatureMetrics]:
        """Build feature-level metrics.

        Returns:
        List[FeatureMetrics]: The feature-level metrics.
        """
        featureimportance_dictionary = {}
        if feature_importance is not None:
            for row in feature_importance:
                featureimportance_dictionary[row[0]] = row[1]

        if not metrics or len(metrics) == 0:
            return []

        output = {}
        for metric in metrics:
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
                "value": metric["metric_value"],
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
            feature_metric = {
                "metricName": metric["metric_name"],
                "metricValue": metric["metric_value"],
            }
            feature_metric = add_value_if_present(
                metric, "threshold_value", feature_metric, "threshold"
            )
            output[feature_name].metrics.append(feature_metric)

            # Add the feature importance metrics
            if len(featureimportance_dictionary) != 0:
                feature_importance_metric = {
                    "metricName": "BaselineFeatureImportance",
                    "metricValue": featureimportance_dictionary[feature_name],
                }
                output[feature_name].metrics.append(feature_importance_metric)

            output[
                feature_name
            ].histogram = f"signals/{self.signal_name}/{feature_name}.histogram.json"

        return list(output.values())
