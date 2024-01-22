# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR header provider."""

import json
import uuid

from ...batch_pool.routing.routing_client import RoutingClient
from ...common.auth.auth_provider import AuthProvider, WorkspaceConnectionAuthProvider
from ...common.auth.token_provider import TokenProvider
from ...common.configuration.configuration import Configuration
from ...common.scoring.generic_scoring_client import HeaderProvider
from ...header_handlers.mir_and_batch_pool_header_handler_factory import MirAndBatchPoolHeaderHandlerFactory


class MirHeaderProvider(HeaderProvider):
    """Defines the MIR header provider."""

    def __init__(
        self,
        auth_provider: AuthProvider,
        configuration: Configuration,
        routing_client: RoutingClient,
        token_provider: TokenProvider,
        additional_headers: str = None
    ):
        self._auth_provider = auth_provider
        self._configuration = configuration
        self._routing_client = routing_client
        self._token_provider = token_provider

        if additional_headers is not None:
            self._additional_headers = json.loads(additional_headers)
        else:
            self._additional_headers = {}


    def get_headers(self) -> dict:
        """Gets the headers from the auth provider and additional headers."""

        # This logic simply adheres to the current behavior.
        # It is likely that a MIR endpoint with API key auth is currently not working.
        # Todo: https://msdata.visualstudio.com/Vienna/_workitems/edit/2897786
        if isinstance(self._auth_provider, WorkspaceConnectionAuthProvider):
            headers = {
                'Content-Type': 'application/json',
                'x-ms-client-request-id': str(uuid.uuid4()),
            }
            headers.update(self._auth_provider.get_auth_headers())
            headers.update(self._additional_headers)
        else:
            header_handler = MirAndBatchPoolHeaderHandlerFactory().get_header_handler(
                configuration = self._configuration,
                routing_client = self._routing_client,
                token_provider=self._token_provider
            )
            headers = header_handler.get_headers(self._additional_headers)
        return headers