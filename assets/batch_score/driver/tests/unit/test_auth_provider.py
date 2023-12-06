# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
from src.batch_score.common.auth.auth_provider import EndpointType, WorkspaceConnectionAuthProvider


@pytest.mark.skip('Need to fix the mock.')
def test_workspace_connection_auth_provider_get_auth_headers(mocker):
    # Arrange
    mocker.patch.object(WorkspaceConnectionAuthProvider, '_get_workspace_connection_by_name', return_value= '{"properties":{"credentials":{"key":"my_value"}}}')

    # Act
    headers = WorkspaceConnectionAuthProvider('my_connection', EndpointType.AOAI).get_auth_headers()

    # Assert
    assert len(headers) == 1
    assert 'api-key' in headers.keys()

