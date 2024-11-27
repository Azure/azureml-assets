# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for content type header provider."""

from .header_provider import HeaderProvider


class ContentTypeHeaderProvider(HeaderProvider):
    """Content type header provider."""

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        return {'Content-Type': 'application/json'}
