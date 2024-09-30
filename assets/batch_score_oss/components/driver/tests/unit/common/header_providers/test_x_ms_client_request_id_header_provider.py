# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for 'x-ms-client-request-id' header provider."""

from src.batch_score_oss.common.header_providers.x_ms_client_request_id_header_provider import (
    XMsClientRequestIdHeaderProvider,
)


def test_get_headers():
    """Test get_headers method."""
    headers = XMsClientRequestIdHeaderProvider().get_headers()
    assert headers.keys() == {'x-ms-client-request-id'}
