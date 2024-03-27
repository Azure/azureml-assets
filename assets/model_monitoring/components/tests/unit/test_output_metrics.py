# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

import pytest
from typing import List
from pyspark.sql import Row

from src.model_monitor_output_metrics.entities.signals.data_drift_signal import (
    DataDriftSignal,
)
from src.model_monitor_output_metrics.entities.signals.data_quality_signal import (
    DataQualitySignal,
)


def _validate_metrics(payload: dict, row: Row):
    assert row["feature_name"] in payload["metrics"]["features"]
    assert (
        payload["metrics"]["features"][row["feature_name"]]["dataType"]
        == row["data_type"]
    )
    assert (
        payload["metrics"]["features"][row["feature_name"]]["metrics"][0]["metricName"]
        == row["metric_name"]
    )
    if payload["signalType"] == "DataQuality":
        _validate_data_quality_metrics(payload, row)
    elif payload["signalType"] == "DataDrift":
        _validate_data_drift_metrics(payload, row)


def _validate_data_drift_metrics(payload: dict, row: Row):
    assert (
        payload["metrics"]["features"][row["feature_name"]]["metrics"][0]["metricValue"]
        == row["metric_value"]
    )


def _validate_data_quality_metrics(payload: dict, row: Row):
    assert (
        payload["metrics"]["features"][row["feature_name"]]["metrics"][0]["baselineValue"]
        == row["baseline_metric_value"]
    )
    assert (
        payload["metrics"]["features"][row["feature_name"]]["metrics"][0]["targetValue"]
        == row["target_metric_value"]
    )


@pytest.mark.unit
class TestOutputMetrics:
    """Test class for data drift output metrics component."""

    def test_generate_datadrift_signal(self):
        """Test generate meta file content for success scenario."""
        signal_metrics: List[Row] = [
            Row(
                feature_name="petal_length",
                metric_value=0.269,
                data_type="Numerical",
                metric_name="WassersteinNormalizedDistance",
            ),
            Row(
                feature_name="sepal_length",
                metric_value=0.746,
                data_type="Numerical",
                metric_name="WassersteinNormalizedDistance",
            ),
            Row(
                feature_name="",
                metric_value=75,
                data_type="",
                metric_name="BaselineRowCount",
            ),
            Row(
                feature_name="",
                metric_value=100,
                data_type="",
                metric_name="TargetRowCount",
            ),
        ]

        feature_importance = [
            Row(
                feature="petal_length",
                metric_value=0.269
            ),
            Row(
                feature="sepal_length",
                metric_value=0.746
            )
        ]

        signal_name = "my-data-drift-signal"
        monitor_name = "my-monitor"

        data_drift_signal = DataDriftSignal(
            monitor_name=monitor_name,
            signal_name=signal_name,
            metrics=signal_metrics,
            baseline_histogram=None,
            target_histogram=None,
            feature_importance=feature_importance,
        )

        payload = data_drift_signal.to_dict()
        print(data_drift_signal.to_dict())
        assert payload["signalName"] == signal_name
        assert payload["signalType"] == "DataDrift"
        assert payload["metrics"]["rowCount"]["baselineCount"] == 75
        assert payload["metrics"]["rowCount"]["targetCount"] == 100
        _validate_metrics(payload, signal_metrics[0])
        _validate_metrics(payload, signal_metrics[1])

    def test_generate_dataquality_signal(self):
        """Test generate meta file content for success scenario."""
        signal_metrics: List[Row] = [
            Row(
                feature_name="petal_length",
                metric_name="DataTypeErrorRate",
                data_type="Categorical",
                baseline_metric_value=None,
                target_metric_value=None,
                group="petal_length",
                group_pivot="",
                metric_value=0,
                threshold_value=0.03
            ),
            Row(
                feature_name="sepal_length",
                metric_name="NullValueRate",
                data_type="Numerical",
                baseline_metric_value=0.0,
                target_metric_value=2.06,
                group="sepal_length",
                group_pivot="",
                metric_value=0,
                threshold_value=0.03
            ),
            Row(
                feature_name="sepal_length",
                metric_name="OutOfBoundsRate",
                data_type="Categoorical",
                baseline_metric_value=10.0,
                target_metric_value=12.06,
                group="sepal_length",
                group_pivot="",
                metric_value=0,
                threshold_value=0.03
            ),
            Row(
                feature_name="",
                metric_name="RowCount",
                data_type="",
                baseline_metric_value=75,
                target_metric_value=100,
                group="",
                group_pivot="",
                metric_value=150,
                threshold_value=0.03
            )
        ]

        feature_importance = [
            Row(
                feature="petal_length",
                metric_value=0.269
            ),
            Row(
                feature="sepal_length",
                metric_value=0.746
            )
        ]

        signal_name = "my-data-quality-signal"
        monitor_name = "my-monitor"

        data_quality_signal = DataQualitySignal(
            monitor_name=monitor_name,
            signal_name=signal_name,
            metrics=signal_metrics,
            feature_importance=feature_importance,
        )

        payload = data_quality_signal.to_dict()
        print(data_quality_signal.to_dict())
        assert payload["signalName"] == signal_name
        assert payload["signalType"] == "DataQuality"
        assert payload["metrics"]["rowCount"]["baselineCount"] == 75
        assert payload["metrics"]["rowCount"]["targetCount"] == 100
        _validate_metrics(payload, signal_metrics[0])
        _validate_metrics(payload, signal_metrics[1])
