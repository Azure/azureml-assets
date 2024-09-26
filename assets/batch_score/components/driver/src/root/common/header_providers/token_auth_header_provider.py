# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for token auth header provider."""

from .header_provider import HeaderProvider
from ..auth.token_provider import TokenProvider


class TokenAuthHeaderProvider(HeaderProvider):
    """Token auth header provider."""

    def __init__(
            self,
            token_provider: TokenProvider,
            token_scope: str):
        """Initialize TokenAuthHeaderProvider."""
        self._token_provider = token_provider
        self._token_scope = token_scope

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        bearer_token = self._token_provider.get_token(scope=self._token_scope)
        return {"Authorization": f"Bearer {bearer_token}"}
