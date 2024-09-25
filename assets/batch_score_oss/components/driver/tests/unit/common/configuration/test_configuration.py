# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for configuration."""

import pytest

from src.batch_score_oss.common.common_enums import ApiType
from src.batch_score_oss.common.configuration.configuration_parser import (
    ConfigurationParser,
)


scoring_url_base = 'https://somebase.microsoft.com'


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.ChatCompletion, None, scoring_url_base, True),
    (ApiType.ChatCompletion, None, None, False),
])
def test_is_chat_completion(api_type, request_path, scoring_url, expected_result):
    """Test is chat completion."""
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_chat_completion() == expected_result


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.Completion, None, scoring_url_base, True),
    (ApiType.Completion, None, None, False),
])
def test_is_completion(api_type, request_path, scoring_url, expected_result):
    """Test is completion."""
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_completion() == expected_result
