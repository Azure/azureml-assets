# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for configuration."""

import pytest

from src.batch_score.batch_pool.routing.routing_client import RoutingClient
from src.batch_score.common import constants
from src.batch_score.common.common_enums import ApiType
from src.batch_score.common.configuration.configuration_parser import (
    ConfigurationParser,
)


scoring_url_base = 'https://somebase.microsoft.com'


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.Completion, constants.DV_EMBEDDINGS_API_PATH, None, True),
    (ApiType.Completion, None, scoring_url_base + constants.DV_EMBEDDINGS_API_PATH, True),
    (ApiType.Embedding, None, scoring_url_base, True),
    (ApiType.Embedding, None, None, False),
    (ApiType.Embedding, constants.DV_CHAT_COMPLETIONS_API_PATH, None, False),
])
def test_is_embeddings(api_type, request_path, scoring_url, expected_result):
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_embeddings() == expected_result


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.Embedding, constants.DV_CHAT_COMPLETIONS_API_PATH, None, True),
    (ApiType.Embedding, None, scoring_url_base + constants.DV_CHAT_COMPLETIONS_API_PATH, True),
    (ApiType.ChatCompletion, None, scoring_url_base, True),
    (ApiType.ChatCompletion, None, None, False),
    (ApiType.ChatCompletion, constants.DV_EMBEDDINGS_API_PATH, None, False),
])
def test_is_chat_completion(api_type, request_path, scoring_url, expected_result):
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
    (ApiType.Embedding, constants.DV_COMPLETION_API_PATH, None, True),
    (ApiType.Embedding, None, scoring_url_base + constants.DV_COMPLETION_API_PATH, True),
    (ApiType.Completion, None, scoring_url_base, True),
    (ApiType.Completion, None, None, False),
    (ApiType.Completion, constants.DV_EMBEDDINGS_API_PATH, None, False),
])
def test_is_completion(api_type, request_path, scoring_url, expected_result):
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_completion() == expected_result


@pytest.mark.parametrize('target_batch_pool, expected_result', [
    ('SaHarA-glObal', True),
    ('sahara-global', True),
    (None, None),
    ('random_pool', False),
])
def test_is_sahara(target_batch_pool, expected_result):
    # Arrange
    routing_client = RoutingClient(
        service_namespace='test_service_namespace',
        target_batch_pool=target_batch_pool,
        header_handler=None,
        request_path='request_path'
    )
    configuration = ConfigurationParser().parse_configuration([])

    # Act & assert
    assert configuration.is_sahara(routing_client) == expected_result


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.Embedding, constants.VESTA_RAINBOW_API_PATH, None, True),
    (ApiType.Embedding, None, scoring_url_base + constants.VESTA_RAINBOW_API_PATH, True),
    (ApiType.Vesta, None, scoring_url_base, True),
    (ApiType.Completion, None, None, False),
    (ApiType.Completion, constants.DV_EMBEDDINGS_API_PATH, None, False),
])
def test_is_vesta(api_type, request_path, scoring_url, expected_result):
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_vesta() == expected_result


@pytest.mark.parametrize('api_type, request_path, scoring_url, expected_result', [
    (ApiType.Embedding, constants.VESTA_CHAT_COMPLETIONS_API_PATH, None, True),
    (ApiType.Embedding, None, scoring_url_base + constants.VESTA_CHAT_COMPLETIONS_API_PATH, True),
    (ApiType.VestaChatCompletion, None, scoring_url_base, True),
    (ApiType.Completion, None, None, False),
    (ApiType.Completion, constants.DV_EMBEDDINGS_API_PATH, None, False),
])
def test_is_vesta_chat_completion(api_type, request_path, scoring_url, expected_result):
    # Arrange
    config_to_parse = ['--api_type', api_type]
    if scoring_url:
        config_to_parse += ['--scoring_url', scoring_url]
    if request_path:
        config_to_parse += ['--request_path', request_path]

    configuration = ConfigurationParser().parse_configuration(config_to_parse)

    # Act & assert
    assert configuration.is_vesta_chat_completion() == expected_result
