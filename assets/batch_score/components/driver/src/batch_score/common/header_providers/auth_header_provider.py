# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for auth header provider."""

from .header_provider import HeaderProvider
from ..auth.auth_provider import AuthProvider


class AuthHeaderProvider(HeaderProvider):
    """Auth header provider."""

    def __init__(self, auth_provider: AuthProvider):
        """Initialize AuthHeaderProvider."""
        self._auth_provider = auth_provider

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        return self._auth_provider.get_auth_headers().copy()
