# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

import pytest
from typing import List
from pyspark.sql import Row

from model_monitor_output_metrics.entities.signals.data_drift_signal import (
    DataDriftSignal,
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
    assert (
        payload["metrics"]["features"][row["feature_name"]]["metrics"][0]["metricValue"]
        == row["metric_value"]
    )


@pytest.mark.unit
class TestDataDriftOutputMetrics:
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
