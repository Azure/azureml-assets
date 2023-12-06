# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from src.batch_score.header_handlers.mir_endpoint_v2_header_handler import MIREndpointV2HeaderHandler
from src.batch_score.common.auth.auth_provider import WorkspaceConnectionAuthProvider


def test_get_headers_no_additional_headers(mocker):
    # Arrange
    handler = MIREndpointV2HeaderHandler('my_connection')
    mocker.patch.object(WorkspaceConnectionAuthProvider, 'get_auth_headers', return_value={'Authorization': 'Bearer 123'})

    # Act
    headers= handler.get_headers()

    # Assert
    assert len(headers) == 3
    _assert_default_headers(headers)
    
def test_get_headers_with_additional_headers(mocker):
    # Arrange
    additional_headers = '{"hello": "world"}'
    handler = MIREndpointV2HeaderHandler('my_connection', additional_headers)
    mocker.patch.object(WorkspaceConnectionAuthProvider, 'get_auth_headers', return_value={'Authorization': 'Bearer 123'})

    # Act
    headers= handler.get_headers()

    # Assert
    assert len(headers) == 4
    assert headers['hello'] == 'world'
    _assert_default_headers(headers)


def _assert_default_headers(headers):
    assert headers['Content-Type'] == 'application/json'
    assert 'Authorization' in headers.keys()
    assert 'x-ms-client-request-id' in headers.keys()
