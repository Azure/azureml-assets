# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for auth header provider."""

from unittest.mock import MagicMock

from src.batch_score.root.common.header_providers.auth_header_provider import AuthHeaderProvider


def test_get_headers():
    """Test get_headers method."""
    auth_headers = {"Authorization": "Bearer token"}
    auth_provider = MagicMock()
    auth_provider.get_auth_headers.return_value = auth_headers
    assert AuthHeaderProvider(auth_provider).get_headers() == auth_headers
