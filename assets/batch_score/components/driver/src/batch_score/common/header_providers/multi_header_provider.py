# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for multi header provider."""

import json

from .header_provider import HeaderProvider


class MultiHeaderProvider:
    """Multi header provider."""

    def __init__(
            self,
            header_providers: "list[HeaderProvider]",
            additional_headers: "dict | str" = None):
        """Initialize MultiHeaderProvider."""
        self._header_providers = header_providers
        self._additional_headers = additional_headers

    def get_headers(self, additional_headers: dict = None) -> dict:
        """Get the headers for requests."""
        headers = {}
        for header_provider in self._header_providers:
            headers.update(header_provider.get_headers())

        headers = self._apply_additional_headers(headers, self._additional_headers)
        headers = self._apply_additional_headers(headers, additional_headers)

        return headers

    def _apply_additional_headers(
            self,
            headers: dict,
            additional_headers: "dict | str") -> dict:
        """Apply additional headers.

        Some parts of the codebase use a dictionary to pass around additional headers, while others use a JSON string.
        This helper ensures the multi header provider safely handles both types.
        """
        if additional_headers is None:
            return headers

        if isinstance(additional_headers, dict):
            headers.update(additional_headers)
            return headers

        headers.update(json.loads(additional_headers))
        return headers
