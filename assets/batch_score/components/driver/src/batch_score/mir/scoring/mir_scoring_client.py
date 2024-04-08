# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR scoring client."""

from .mir_http_response_handler import MirHttpResponseHandler
from ...common.configuration.configuration import Configuration
from ...common.scoring.generic_scoring_client import GenericScoringClient
from ...common.header_providers.header_provider import HeaderProvider
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler


class MirScoringClient(GenericScoringClient):
    """Defines the MIR scoring client."""

    def __init__(
            self,
            header_provider: HeaderProvider,
            configuration: Configuration,
            tally_handler: TallyFailedRequestHandler = None,
            scoring_url: str = None):
        """Initialize MirScoringClient.

        The scoring_url parameter is optional. If not provided, the scoring_url from the configuration will be used.
        """
        super().__init__(
            header_provider=header_provider,
            http_response_handler=MirHttpResponseHandler(tally_handler),
            scoring_url=scoring_url or configuration.scoring_url)
