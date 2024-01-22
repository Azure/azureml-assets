# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for aoai  header provider."""

from src.batch_score.aoai.scoring.aoai_header_provider import AoaiHeaderProvider
from src.batch_score.common.auth.auth_provider import EndpointType, WorkspaceConnectionAuthProvider

def test_get_headers_no_additional_headers(mocker):
    # Arrange
    auth_provider = WorkspaceConnectionAuthProvider(connection_name='test_key', endpoint_type=EndpointType.AOAI)
    header_provider = AoaiHeaderProvider(auth_provider=auth_provider, additional_headers=None)
    mocker.patch.object(WorkspaceConnectionAuthProvider, 'get_auth_headers', return_value={'Authorization': 'Bearer 123'})

    # Act
    headers = header_provider.get_headers()

    # Assert
    assert len(headers) == 3
    _assert_default_headers(headers)


def test_get_header_with_additional_headers(mocker):
    # Arrange
    auth_provider = WorkspaceConnectionAuthProvider(connection_name='test_key', endpoint_type=EndpointType.AOAI)
    header_provider = AoaiHeaderProvider(auth_provider=auth_provider, additional_headers='{"hello": "world"}')
    mocker.patch.object(WorkspaceConnectionAuthProvider, 'get_auth_headers', return_value={'Authorization': 'Bearer 123'})

    # Act
    headers = header_provider.get_headers()

    # Assert
    assert len(headers) == 4
    assert headers['hello'] == 'world'
    _assert_default_headers(headers)


def _assert_default_headers(headers):
    assert headers['Content-Type'] == 'application/json'
    assert 'x-ms-client-request-id' in headers.keys()