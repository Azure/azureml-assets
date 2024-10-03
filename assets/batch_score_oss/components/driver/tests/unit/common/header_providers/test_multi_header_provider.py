# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for multi header provider."""

from src.batch_score_oss.common.header_providers.content_type_header_provider import (
    ContentTypeHeaderProvider,
)
from src.batch_score_oss.common.header_providers.multi_header_provider import MultiHeaderProvider
from src.batch_score_oss.common.header_providers.traffic_group_header_provider import (
    TrafficGroupHeaderProvider,
)


def test_get_headers():
    """Test get_headers method."""
    multi_provider = MultiHeaderProvider(
        header_providers=[
            ContentTypeHeaderProvider(),
            TrafficGroupHeaderProvider(),
        ],
        additional_headers={"header1": "value1"})

    headers = multi_provider.get_headers(additional_headers='{"header2": "value2"}')

    assert headers == {
        "Content-Type": "application/json",
        "azureml-model-group": "batch",
        "azureml-collect-request": "false",
        "azureml-inferencing-offer-name": "azureml_vanilla",
        "header1": "value1",
        "header2": "value2",
    }
