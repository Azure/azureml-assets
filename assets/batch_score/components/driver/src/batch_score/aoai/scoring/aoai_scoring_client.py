# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for AOAI scoring client."""

from .aoai_response_handler import AoaiHttpResponseHandler
from ...common.header_providers.header_provider import HeaderProvider
from ...common.scoring.generic_scoring_client import GenericScoringClient
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler


class AoaiScoringClient(GenericScoringClient):
    """Defines the AOAI scoring client."""

    def __init__(
            self,
            header_provider: HeaderProvider,
            scoring_url: str = None,
            tally_handler: TallyFailedRequestHandler = None):
        """Initialize AoaiScoringClient."""
        super().__init__(
            header_provider=header_provider,
            http_response_handler=AoaiHttpResponseHandler(tally_handler),
            scoring_url=scoring_url)
