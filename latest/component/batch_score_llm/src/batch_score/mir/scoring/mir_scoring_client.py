# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR scoring client."""

import aiohttp

from ...common.configuration.configuration import Configuration
from ...common.scoring.generic_scoring_client import GenericScoringClient
from ...common.header_providers.header_provider import HeaderProvider
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import ScoringResult
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from .mir_http_response_handler import MirHttpResponseHandler


class MirScoringClient:
    """Defines the MIR scoring client."""

    def __init__(
        self,
        header_provider: HeaderProvider,
        configuration: Configuration,
        tally_handler: TallyFailedRequestHandler = None,
        scoring_url: str = None,
    ):
        """Initialize MirScoringClient.

        The scoring_url parameter is optional. If not provided, the scoring_url from the configuration will be used.
        """
        self._generic_scoring_client = GenericScoringClient(
            header_provider=header_provider,
            http_response_handler=MirHttpResponseHandler(tally_handler=tally_handler),
            scoring_url=scoring_url or configuration.scoring_url)

    async def score_once(
        self,
        session: aiohttp.ClientSession,
        scoring_request: ScoringRequest,
        timeout: aiohttp.ClientTimeout = None,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score a single request until terminal status is reached."""
        # TODO: Validate if the error logging in generic client has all the properties that
        # previous scoring client logs.
        return await self._generic_scoring_client.score(
            session=session,
            scoring_request=scoring_request,
            timeout=timeout)

    def validate_auth(self):
        """Validate the auth by sending dummy request to the scoring url."""
        self._generic_scoring_client.validate_auth()
