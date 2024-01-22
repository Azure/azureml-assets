# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for auth provider factory."""

import pytest
import azureml.core
from src.batch_score.common.auth.auth_provider import (
    ApiKeyAuthProvider,
    IdentityAuthProvider,
    WorkspaceConnectionAuthProvider
)
from src.batch_score.common.auth.auth_provider_factory import AuthProviderFactory
from src.batch_score.common.configuration.configuration_parser import ConfigurationParser


@pytest.mark.parametrize('authentication_type, expected_auth_type', [
    ('api_key', ApiKeyAuthProvider),
    ('managed_identity', IdentityAuthProvider),
    ('azureml_workspace_connection', WorkspaceConnectionAuthProvider)
])
def test_get_auth_provider(mocker, authentication_type, expected_auth_type):
    # Arrange
    configuration = ConfigurationParser().parse_configuration([
        '--authentication_type', authentication_type,
        '--scoring_url', 'hello.openai.azure.com',
        '--api_key_name', 'test_api_key',
        '--connection_name', 'my_connection'
    ])
    mocker.patch("azureml.core.Run.get_context", return_value=azureml.core.Run)
    mocker.patch("azureml.core.Run.get_secret", return_value='mysecret')

    # Act
    auth_provider = AuthProviderFactory().get_auth_provider(configuration)

    # Assert
    assert isinstance(auth_provider, expected_auth_type)
