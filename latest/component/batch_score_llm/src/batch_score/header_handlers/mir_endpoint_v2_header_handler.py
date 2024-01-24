# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MIR endpoint V2 header handler."""

import uuid

from .header_handler import HeaderHandler
from ..common.auth.auth_provider import EndpointType, WorkspaceConnectionAuthProvider


class MIREndpointV2HeaderHandler(HeaderHandler):
    """MIR endpoint V2 header handler."""

    def __init__(self, connection_name: str, additional_headers: str = None) -> None:
        """Initialize MIREndpointV2HeaderHandler."""
        super().__init__(
            token_provider=None,
            additional_headers=additional_headers)
        self.__connection_name = connection_name

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get the headers for MIR endpoint requests."""
        headers = {
            'Content-Type': 'application/json',
            'x-ms-client-request-id': str(uuid.uuid4())
        }

        auth_headers = WorkspaceConnectionAuthProvider(self.__connection_name, EndpointType.MIR).get_auth_headers()
        headers.update(auth_headers)
        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
