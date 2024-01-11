# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Azure Monitor Metric Publisher component."""

from datetime import datetime

from pyspark.sql import Row
import pytest

from model_monitor_azmon_metric_publisher.azure_monitor_metric_publisher import (
    to_metric_payload,
    extract_location_from_aml_endpoint_str,
)

test_monitor_name = "test_monitor"
test_signal_name = "test_signal"
test_window_start = datetime(year=2023, month=1, day=1, hour=8)
test_window_end = datetime(year=2023, month=1, day=8, hour=8)
test_location = "eastus"
test_ws_resource_uri = "/subscriptions/ea4faa5b-5e44-4236-91f6-5483d5b17d14/resourceGroups/model-monitoring-canary-rg"\
                        + "/providers/Microsoft.MachineLearningServices/workspaces/model-monitoring-canary-ws/"

successful_test_cases = [
    # empty or no group dimension
    Row(
        group="group_1",
        group_dimension=None,
        metric_name="num_calls",
        metric_value=71.0,
        threshold_value=100.0,
    ),
    Row(
        group="group_2",
        group_dimension="",
        metric_name="num_calls",
        metric_value=72.0,
        threshold_value=100.0,
    ),
    # constant group dimension
    Row(
        group="group_3",
        group_dimension="Aggregate",
        metric_name="num_calls",
        metric_value=73.0,
        threshold_value=100.0,
    ),
    # single level group dimension
    Row(
        group="group_1",
        group_dimension="sepal_length",
        metric_name="num_calls_with_status_code_429",
        metric_value=35.0,
        threshold_value=10.0,
    ),
    # no threshold value
    Row(
        group="group_2",
        group_dimension="sepal_length",
        metric_name="num_calls_with_status_code_429",
        metric_value=63.0,
        threshold_value=None,
    ),
]


@pytest.mark.unit1
class TestAzureMonitorMetricPublisher:
    """Test class for Azure monitor metric publisher."""

    @pytest.mark.parametrize("input_metric", successful_test_cases)
    def test_to_metric_payload_success(self, input_metric):
        """Test to_metric_payload method for happy paths."""
        payload = to_metric_payload(
            input_metric,
            test_monitor_name,
            test_signal_name,
            test_window_start,
            test_window_end)

        assert payload is not None
        assert payload["time"] is not None

        baseData = payload["data"]["baseData"]
        assert baseData is not None
        assert baseData["dimNames"] is not None
        assert baseData["series"][0]["dimValues"] is not None
        assert input_metric["metric_name"] == baseData["metric"]
        assert input_metric["metric_value"] == baseData["series"][0]["min"]
        assert input_metric["metric_value"] == baseData["series"][0]["max"]
        assert input_metric["metric_value"] == baseData["series"][0]["sum"]

    @pytest.mark.parametrize("aml_service_endpoint, expected_location", [
        ("https://eastus.api.azureml.ms", "eastus"),
        ("https://eastus2.api.azureml.ms", "eastus2"),
        ("https://eastus2euap.api.azureml.ms", "eastus2euap"),
    ])
    def test_extract_location_success(self, aml_service_endpoint, expected_location):
        """Test extract_location_from_aml_endpoint_str method for happy paths."""
        location = extract_location_from_aml_endpoint_str(aml_service_endpoint)
        assert expected_location == location
