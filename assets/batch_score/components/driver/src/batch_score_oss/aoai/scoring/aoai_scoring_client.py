# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for AOAI scoring client."""

from .aoai_response_handler import AoaiHttpResponseHandler
from ...common.configuration.configuration import Configuration
from ...common.header_providers.header_provider import HeaderProvider
from ...common.scoring.generic_scoring_client import GenericScoringClient
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler


class AoaiScoringClient(GenericScoringClient):
    """Defines the AOAI scoring client."""

    def __init__(
            self,
            header_provider: HeaderProvider,
            configuration: Configuration,
            tally_handler: TallyFailedRequestHandler = None):
        """Initialize AoaiScoringClient.

        :param header_provider: header provider for request creation
        :param header_provider: HeaderProvider
        :param configuration: Batch score configuration
        :type configuration: Configuration
        :param tally_handler: tallyHandler instance created on batch scoring configuration to tally failed requests
        :type tally_handler: Optional|TallyFailedRequestHandler
        """
        super().__init__(
            header_provider=header_provider,
            http_response_handler=AoaiHttpResponseHandler(tally_handler, configuration),
            scoring_url=configuration.scoring_url)
