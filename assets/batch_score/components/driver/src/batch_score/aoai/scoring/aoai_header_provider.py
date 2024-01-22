# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for AOAI header provider."""

import json
import uuid

from ...common.auth.auth_provider import AuthProvider
from ...common.scoring.generic_scoring_client import HeaderProvider


class AoaiHeaderProvider(HeaderProvider):
    """Defines the AOAI header provider."""

    def __init__(
        self,
        auth_provider: AuthProvider,
        additional_headers: str = None,
    ):
        self._auth_provider = auth_provider

        if additional_headers is not None:
            self._additional_headers = json.loads(additional_headers)
        else:
            self._additional_headers = {}


    def get_headers(self) -> dict:
        """Gets the headers from the auth provider and additional headers."""
        headers = {
            'Content-Type': 'application/json',
            'x-ms-client-request-id': str(uuid.uuid4()),
        }
        headers.update(self._auth_provider.get_auth_headers())
        headers.update(self._additional_headers)

        return headers