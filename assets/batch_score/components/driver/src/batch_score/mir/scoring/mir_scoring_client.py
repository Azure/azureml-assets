# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR scoring client."""

import aiohttp

from ...batch_pool.routing.routing_client import RoutingClient
from ...common.auth.auth_provider import AuthProvider
from ...common.auth.token_provider import TokenProvider
from ...common.configuration.configuration import Configuration
from ...common.scoring.generic_scoring_client import GenericScoringClient
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import ScoringResult
from ...common.telemetry import logging_utils as lu
from .mir_header_provider import MirHeaderProvider
from .mir_http_response_handler import MirHttpResponseHandler


class MirScoringClient:
    """Defines the MIR scoring client."""

    def __init__(
        self,
        auth_provider: AuthProvider,
        configuration: Configuration,
        routing_client: RoutingClient,
        token_provider: TokenProvider,
        additional_headers: str = None
    ):
        """Initialize MirScoringClient."""
        header_provider = MirHeaderProvider(
            auth_provider=auth_provider,
            configuration=configuration,
            routing_client=routing_client,
            token_provider=token_provider,
            additional_headers=additional_headers
        )

        self._generic_scoring_client = GenericScoringClient(
            header_provider=header_provider,
            http_response_handler=MirHttpResponseHandler(),
            scoring_url=configuration.scoring_url,
        )

    async def score_once(
        self,
        session: aiohttp.ClientSession,
        scoring_request: ScoringRequest,
        timeout: aiohttp.ClientTimeout = None,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score a single request until terminal status is reached."""

        # Timeout can be None. See `timeout_utils.get_next_retry_timeout` for more info on why.
        if timeout is None:
            timeout = session.timeout

        # Adding this to not break any existing logging.
        lu.get_logger().debug(
            f"Worker_id: {worker_id}, internal_id: {scoring_request.internal_id}, Timeout: {timeout.total}s")

        # TODO: Validate if the error logging in generic client has all the properties that
        # previous scoring client logs.
        return await self._generic_scoring_client.score(
            session=session,
            scoring_request=scoring_request,
            timeout=timeout,
        )
