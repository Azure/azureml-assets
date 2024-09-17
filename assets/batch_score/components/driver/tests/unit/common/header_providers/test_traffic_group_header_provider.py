# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for traffic group header provider."""

from src.batch_score_oss.root.common.header_providers.traffic_group_header_provider import (
    TrafficGroupHeaderProvider,
)


def test_get_headers():
    """Test get_headers method."""
    assert TrafficGroupHeaderProvider().get_headers() == {
        "azureml-model-group": "batch",
        "azureml-collect-request": "false",
        "azureml-inferencing-offer-name": "azureml_vanilla",
    }
