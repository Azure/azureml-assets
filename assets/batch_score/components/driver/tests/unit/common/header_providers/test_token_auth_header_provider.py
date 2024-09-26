# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for token auth header provider."""

from unittest.mock import MagicMock

from src.batch_score.root.common.header_providers.token_auth_header_provider import (
    TokenAuthHeaderProvider,
)


def test_get_headers():
    """Test get_headers method."""
    token_provider = MagicMock()
    token_scope = "kaleidoscope"
    token = "totoken"

    def mock_get_token(scope):
        assert scope == token_scope
        return token

    token_provider.get_token = mock_get_token
    headers = TokenAuthHeaderProvider(token_provider, token_scope).get_headers()
    assert headers == {"Authorization": f"Bearer {token}"}
