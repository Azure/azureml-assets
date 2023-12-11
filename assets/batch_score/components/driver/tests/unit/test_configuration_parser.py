# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for configuration parser."""

import pytest

import src.batch_score.common.constants as constants
import src.batch_score.main as main
from src.batch_score.common.auth.auth_provider import EndpointType
from src.batch_score.common.configuration.configuration_parser import (
    ConfigurationParser,
)


def test_success_defaults():
    """Test success defaults."""
    # Act
    configuration = ConfigurationParser().parse_configuration()

    # Assert
    assert configuration.batch_size_per_request == 1
    assert configuration.additional_headers is None


def test_success_scoring_url_copied_to_endpoint_url():
    """Test success scoring url copied to endpoint url."""
    # Act
    scoring_url = "https://non-existent-endpoint.centralus.inference.ml.azure.com/score"
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])

    # Assert
    assert configuration.online_endpoint_url == scoring_url


@pytest.mark.parametrize(
    'api_type, segment_large_requests',
    [
        (constants.CHAT_COMPLETION_API_TYPE, 'disabled'),
        (constants.COMPLETION_API_TYPE, 'enabled'),
        (constants.EMBEDDINGS_API_TYPE, 'disabled'),
        (constants.VESTA_API_TYPE, 'disabled'),
        (constants.VESTA_CHAT_COMPLETION_API_TYPE, 'disabled'),
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


@pytest.mark.parametrize('request_path', [
    "v1/engines/davinci/completions",
    "v1/engines/davinci/chat/completions",
    "v1/engines/davinci/embeddings",
    "v1/rainbow"
])
@pytest.mark.parametrize('header, pool, expected_values', [
    ("{}", "pool", {"azureml-collect-request": 'false', "azureml-inferencing-offer-name": 'azureml_vanilla'}),
    ("{}", None, {}),
    ('{"something":"else"}',
     "pool",
     {"azureml-collect-request": 'false', "something": "else", "azureml-inferencing-offer-name": 'azureml_vanilla'}),
    ('{"something":"else"}', None, {"something": "else"}),
    ('{"Azureml-collect-request":"False"}',
     "pool",
     {"Azureml-collect-request": "False", "azureml-inferencing-offer-name": 'azureml_vanilla'}),
    ('{"Azureml-collect-request":"False"}', None, {"Azureml-collect-request": "False"}),
    ('{"Azureml-collect-request": true}',
     "pool",
     {"Azureml-collect-request": True, "azureml-inferencing-offer-name": 'azureml_vanilla'}),
    ('{"Azureml-collect-request": true}', None, {"Azureml-collect-request": True}),
    ('{"Azureml-inferencing-Offer-naMe": "another_offering"}',
     None,
     {"Azureml-inferencing-Offer-naMe": "another_offering"}),
    ('{"azureml-collect-request": "true","some":"else"}',
     "pool",
     {"azureml-collect-request": "true", "some": "else", "azureml-inferencing-offer-name": 'azureml_vanilla'}),
    ('{"Azureml-inferencing-offer-name": "my_custom_offering","some":"else"}',
     "pool",
     {"azureml-collect-request": "false", "some": "else", "Azureml-inferencing-offer-name": 'my_custom_offering'})
])
def test_success_additional_header(header, pool, expected_values, request_path):
    """Test success additional header."""
    # TODO: headers with booleans fail during session.post.
    #  Prevent users from providing additional_headers that json.loads with boolean values.
    # Act
    args = ["--additional_headers", header, "--request_path", request_path]
    if pool:
        args += ["--batch_pool", pool]

    configuration = ConfigurationParser().parse_configuration(args)
    main.configuration = configuration
    header_handler = main.setup_header_handler(None, None)

    # Assert
    for key, value in expected_values.items():
        assert header_handler._additional_headers[key] == value
    if "azureml-collect-request" not in expected_values.keys():
        assert "azureml-collect-request" not in header_handler._additional_headers


def test_success_batch_size_one():
    """Test success batch size one case."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--batch_size_per_request", '1'])

    # Assert
    assert configuration.batch_size_per_request == 1


def test_success_batch_size_greater_than_one_with_embeddings_api():
    """Test success batch size greater than one with embeddings api case."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--request_path",
                                                               "v1/engines/davinci/embeddings",
                                                               "--batch_size_per_request",
                                                               '20'])

    # Assert
    assert configuration.batch_size_per_request == 20


def test_invalid_batch_size_zero():
    """Test invalid batch size zero case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--batch_size_per_request", '0'])

    # Assert
    assert "The optional parameter 'batch_size_per_request' cannot be less than 1." \
           " Valid range is 1-2000." in str(excinfo.value)


def test_invalid_batch_size_greater_than_max_with_embeddings_api():
    """Test invalid batch size greater than max with embeddings api case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--batch_size_per_request", '2001'])

    # Assert
    assert "The optional parameter 'batch_size_per_request' cannot be greater than 2000." \
           " Valid range is 1-2000." in str(excinfo.value)


def test_invalid_batch_size_greater_than_one_with_completions_api():
    """Test invalid batch size greater than one with completions api case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--request_path",
                                                       "v1/engines/davinci/chat/completions",
                                                       "--batch_size_per_request",
                                                       '2'])

    # Assert
    assert "The optional parameter 'batch_size_per_request' is only allowed to be greater than 1 for the Embeddings " \
           "API." in str(excinfo.value)


@pytest.mark.parametrize('request_path', [
    "v1/engines/davinci/chat/completions",
    "v1/engines/davinci/embeddings",
    "v1/rainbow",
])
def test_invalid_segment_large_requests_with_unsupported_api(request_path):
    """Test invalid segment large request with unsupported api case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--request_path",
                                                       request_path,
                                                       "--segment_large_requests", 'enabled'])

    # Assert
    assert "The optional parameter 'segment_large_requests' is supported only with the Completion API." \
           "Please set 'segment_large_requests' to 'disabled' or remove it from the configuration." \
           in str(excinfo.value)


def test_invalid_request_path_with_online_endpoint_url():
    """Test invalid request path with online endpoint url case."""
    # Act
    with pytest.raises(ValueError) as excinfo:
        _ = ConfigurationParser().parse_configuration(["--request_path",
                                                       "v1/engines/ada/embeddings",
                                                       "--online_endpoint_url",
                                                       'test.azure'])

    # Assert
    assert "The optional parameter 'online_endpoint_url' is not allowed in combination with 'request_path'. " \
           "Please put the entire scoring url in the `online_endpoint_url` parameter and remove 'request_path'." \
           in str(excinfo.value)


@pytest.mark.parametrize("scoring_url, expected_is_aoai_endpoint", [
    ("https://batchscore.openai.azure.com/openai"
     "/deployments/turbo/completions?api-version=2023-03-15-preview", True),
    ("https://batchscore.api.cognitive.microsoft.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", True),
    ("https://batchscore.cognitiveservices.azure.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", True),
    ("https://batchscore.azure.com/openai"
     "/deployments/turbo/completions?api-version=2023-03-15-preview", False),
    ("https://llama-completion.eastus2.inference.ai.azure.com/v1/completions", False)
])
def test_is_aoai_endpoint(scoring_url, expected_is_aoai_endpoint):
    """Test is AOAI endpoint."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])

    # Assert
    assert configuration.is_aoai_endpoint() == expected_is_aoai_endpoint


@pytest.mark.parametrize("scoring_url, expected_is_serverless_endpoint", [
    ("https://batchscore.openai.azure.com/openai"
     "/deployments/turbo/completions?api-version=2023-03-15-preview", False),
    ("https://batchscore.api.cognitive.microsoft.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", False),
    ("https://batchscore.cognitiveservices.azure.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", False),
    ("https://batchscore.azure.com/openai"
     "/deployments/turbo/completions?api-version=2023-03-15-preview", False),
    ("https://llama-completion.eastus2.inference.ai.azure.com/v1/completions", True)
])
def test_is_serverless_endpoint(scoring_url, expected_is_serverless_endpoint):
    """Test is serverless endpoint."""
    # Act
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])

    # Assert
    assert configuration.is_serverless_endpoint() == expected_is_serverless_endpoint


@pytest.mark.parametrize('scoring_url, expected_endpoint_type', [
    ("https://batchscore.openai.azure.com/openai"
     "/deployments/turbo/completions?api-version=2023-03-15-preview", EndpointType.AOAI),
    ("https://batchscore.api.cognitive.microsoft.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", EndpointType.AOAI),
    ("https://batchscore.cognitiveservices.azure.com/openai"
     "/deployments/turbo/chat/completions?api-version=2023-03-15-preview", EndpointType.AOAI),
    ("https://batchscore.inference.ml.azure.com/v1/completions", EndpointType.MIR),
    ("https://llama-completion.eastus2.inference.ai.azure.com/v1/completions", EndpointType.Serverless)
])
def test_get_endpoint_type(scoring_url, expected_endpoint_type):
    """Test get endpoint type."""
    # Arrange
    configuration = ConfigurationParser().parse_configuration(["--scoring_url", scoring_url])
    # Act & Assert
    assert configuration.get_endpoint_type() == expected_endpoint_type
