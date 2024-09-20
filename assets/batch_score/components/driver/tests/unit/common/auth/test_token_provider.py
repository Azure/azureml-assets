# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for token provider."""

import src.batch_score.common.auth.token_provider as tp


def test_get_token(mock_get_logger, make_token_provider, mock__credentials_get_tokens):
    """Test get token."""
    token_provider: tp.TokenProvider = make_token_provider(client_id="fake_client_id")

    token_provider.get_token(tp.TokenProvider.SCOPE_AML)
    assert mock_get_logger.debug.called

    assert token_provider
