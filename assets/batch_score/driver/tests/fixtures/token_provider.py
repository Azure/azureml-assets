# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from src.batch_score.common.auth.token_provider import TokenProvider


@pytest.fixture
def make_token_provider():
    def make(client_id=None, token_file_path: str = None) -> TokenProvider:
        token_provider = TokenProvider(
            client_id=client_id,
            token_file_path=token_file_path)

        return token_provider

    return make


@pytest.fixture
def make_access_token():
    def make(token: str = "MOCK_TOKEN"):
        class FakeAccessToken():
            def __init__(self, token: str) -> None:
                self.token = token

        return FakeAccessToken(token)

    return make


@pytest.fixture
def mock__credentials_get_tokens(monkeypatch, make_access_token):
    requested_scopes = []

    def _get_token(self, scope):
        requested_scopes.append(scope)

        return make_access_token()

    monkeypatch.setattr("azure.identity._credentials.managed_identity.ManagedIdentityCredential.get_token", _get_token)
    return requested_scopes
