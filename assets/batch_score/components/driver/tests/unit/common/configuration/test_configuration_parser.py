# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for configuration parser."""

import pytest

from src.batch_score_oss.root.common import constants
from src.batch_score_oss.root.common.common_enums import ApiType
from src.batch_score_oss.root.common.auth.auth_provider import EndpointType
from src.batch_score_oss.root.common.configuration.configuration_parser import (
    ConfigurationParser,
)


def test_success_defaults():
    """Test success defaults."""
    # Act
    configuration = ConfigurationParser().parse_configuration()

    # Assert
    assert configuration.batch_size_per_request == 1
    assert configuration.additional_headers is None


def test_success_online_endpoint_url_copied_to_scoring_url():
    """Test success scoring url copied to endpoint url."""
    # Act
    online_endpoint_url = "https://non-existent-endpoint.centralus.inference.ml.azure.com/score"
    configuration = ConfigurationParser().parse_configuration(["--online_endpoint_url", online_endpoint_url])

    # Assert
    assert configuration.scoring_url == online_endpoint_url


@pytest.mark.parametrize(
    'api_type, segment_large_requests',
    [
        (ApiType.ChatCompletion, 'disabled'),
        (ApiType.Completion, 'enabled'),
    ],
)
def test_success_set_default_for_segment_large_requests(api_type, segment_large_requests):
    """Test success set default segment large requests case."""
    # Act
    scoring_url = "https://non-existent-endpoint.centralus.inference.ml.azure.com/score"
    configuration = ConfigurationParser().parse_configuration([
        "--api_type", api_type,
        "--scoring_url", scoring_url,
    ])

    # Assert
    assert configuration.segment_large_requests == segment_large_requests


def test_success_batch_size_one():
    """Test success batch size one case."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--batch_size_per_request", '1'])

    # Assert
    assert configuration.batch_size_per_request == 1


def test_invalid_batch_size_zero():
    """Test invalid batch size zero case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--batch_size_per_request", '0'])

    # Assert
    assert "The optional parameter 'batch_size_per_request' cannot be less than 1." \
           " Valid range is 1-2000." in str(excinfo.value)


def test_invalid_batch_size_greater_than_one_with_completions_api():
    """Test invalid batch size greater than one with completions api case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--request_path",
                                                       constants.OSS_CHAT_COMPLETIONS_API_PATH,
                                                       "--batch_size_per_request", '2'])

    # Assert
    assert "The optional parameter 'batch_size_per_request' is only allowed to be greater than 1 for the Embeddings " \
           "API." in str(excinfo.value)


@pytest.mark.parametrize('request_path', [
    "v1/chat/completions",
])
def test_invalid_segment_large_requests_with_unsupported_api(request_path):
    """Test invalid segment large request with unsupported api case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--request_path",
                                                       request_path,
                                                       "--segment_large_requests",
                                                       'enabled'])

    # Assert
    assert "The optional parameter 'segment_large_requests' is supported only with the Completion API. " \
           "Please set 'segment_large_requests' to 'disabled' or remove it from the configuration." \
           in str(excinfo.value)


@pytest.mark.parametrize("scoring_url, expected_is_serverless_endpoint", [
    ("https://llama-completion.eastus2.inference.ai.azure.com/v1/completions", True)
])
def test_is_serverless_endpoint(scoring_url, expected_is_serverless_endpoint):
    """Test is serverless endpoint."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])

    # Assert
    assert configuration.is_serverless_endpoint() == expected_is_serverless_endpoint


@pytest.mark.parametrize('scoring_url, expected_endpoint_type', [
    ("https://llama-completion.eastus2.inference.ai.azure.com/v1/completions", EndpointType.Serverless)
])
def test_get_endpoint_type(scoring_url, expected_endpoint_type):
    """Test get endpoint type."""
    # Arrange
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])
    # Act & Assert
    assert configuration.get_endpoint_type() == expected_endpoint_type
