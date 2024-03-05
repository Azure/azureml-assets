# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for multi header provider."""

from .header_provider import HeaderProvider


class MultiHeaderProvider:
    """Multi header provider."""

    def __init__(
            self,
            header_providers: "list[HeaderProvider]",
            additional_headers: dict = None):
        """Initialize MultiHeaderProvider."""
        self._header_providers = header_providers
        self._additional_headers = additional_headers

    def get_headers(self, additional_headers: dict = None) -> dict:
        """Get the headers for requests."""
        headers = {}
        for header_provider in self._header_providers:
            headers.update(header_provider.get_headers())

        if self._additional_headers:
            headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
