# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for x-ms-client-request-id header provider."""

import uuid

from .header_provider import HeaderProvider


class XMsClientRequestIdHeaderProvider(HeaderProvider):
    """Content type header provider."""

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        return {'x-ms-client-request-id': str(uuid.uuid4())}
