# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for MIR header provider."""

import pytest

from src.batch_score.common.auth.auth_provider import IdentityAuthProvider, WorkspaceConnectionAuthProvider
from src.batch_score.mir.scoring.mir_header_provider import MirHeaderProvider
from src.batch_score.header_handlers.mir_and_batch_pool_header_handler_factory import (
    MirAndBatchPoolHeaderHandlerFactory
)
from src.batch_score.header_handlers.open_ai import OpenAIHeaderHandler


@pytest.mark.parametrize('has_additional_headers', [True, False])
def test_get_headers_for_connection_auth_provider(has_additional_headers, mocker):
    # Arrange
    auth_provider = WorkspaceConnectionAuthProvider(connection_name='test', endpoint_type='MIR')
    expected_header_from_auth_provider = {'my_auth': '123'}
    additional_headers = '{"additional_header": "true"}'
    mocker.patch.object(WorkspaceConnectionAuthProvider, 'get_auth_headers',
                        return_value=expected_header_from_auth_provider)

    if has_additional_headers:
        header_provider = MirHeaderProvider(auth_provider, None, None, None, additional_headers=additional_headers)
    else:
        header_provider = MirHeaderProvider(auth_provider, None, None, None)

    # Act
    actual_headers = header_provider.get_headers()

    # Assert
    assert len(actual_headers) == (4 if has_additional_headers else 3)
    assert actual_headers['Content-Type'] == 'application/json'
    assert 'x-ms-client-request-id' in actual_headers.keys() 
    assert expected_header_from_auth_provider.items() <= actual_headers.items()

    if has_additional_headers:
        assert actual_headers['additional_header'] == 'true'


def test_get_headers_uses_mir_header_handler(mocker):
    # Arrange
    auth_provider = IdentityAuthProvider(use_user_identity=False)
    mocker.patch.object(MirAndBatchPoolHeaderHandlerFactory, 'get_header_handler', return_value=OpenAIHeaderHandler)
    expected_headers = {'hello': 'world'}
    mocker.patch.object(OpenAIHeaderHandler, 'get_headers', return_value=expected_headers)
    header_provider = MirHeaderProvider(auth_provider, None, None, None)

    # Act
    actual_headers = header_provider.get_headers()

    # Assert
    assert actual_headers == expected_headers
