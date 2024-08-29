# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for scoring client factory."""

import json

from .null_endpoint_scoring_client import NullEndpointScoringClient
from ...aoai.scoring.aoai_response_handler import AoaiHttpResponseHandler
from ...aoai.scoring.aoai_scoring_client import AoaiScoringClient
from ...common.auth.auth_provider_factory import AuthProviderFactory
from ...common.auth.token_provider import TokenProvider
from ...common.common_enums import EndpointType
from ...common.configuration.configuration import Configuration
from ...common.configuration.metadata import Metadata
from ...common.header_providers.header_provider import HeaderProvider
from ...common.header_providers.header_provider_factory import HeaderProviderFactory
from ...common.scoring.scoring_client import ScoringClient
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.logging_utils import (
    get_logger
)


class ScoringClientFactory:
    """Defines the scoring client factory."""

    def setup_scoring_client(
        self,
        configuration: Configuration,
        metadata: Metadata,
        token_provider: TokenProvider
    ) -> ScoringClient:
        """Get the scoring client for the given configuration."""
        auth_provider = AuthProviderFactory().get_auth_provider(configuration)
        additional_headers = self._get_additional_headers(configuration)
        header_provider = HeaderProviderFactory().get_header_provider_for_scoring(
            auth_provider=auth_provider,
            configuration=configuration,
            metadata=metadata,
            token_provider=token_provider,
            additional_headers=additional_headers)
        tally_handler = _setup_tally_handler(configuration)
        endpoint_type = configuration.get_endpoint_type()

        if endpoint_type in [EndpointType.AOAI, EndpointType.Serverless]:
            get_logger().info("Using LLM scoring client.")
            return _setup_aoai_scoring_client(
                configuration=configuration,
                header_provider=header_provider,
                tally_handler=tally_handler)

        get_logger().info("Using null endpoint scoring client.")
        return _setup_null_endpoint_scoring_client(
            configuration=configuration,
            tally_handler=tally_handler)

    def _get_additional_headers(self, configuration):
        if configuration.additional_headers is None:
            headers = {}
        elif isinstance(configuration.additional_headers, dict):
            headers = configuration.additional_headers.copy()
        else:
            headers = json.loads(configuration.additional_headers)

        return headers


def _setup_aoai_scoring_client(
        configuration: Configuration,
        header_provider: HeaderProvider,
        tally_handler: TallyFailedRequestHandler) -> ScoringClient:
    """Create an instance of an AOAI scoring client."""
    scoring_client = AoaiScoringClient(
        header_provider=header_provider,
        configuration=configuration,
        tally_handler=tally_handler)

    scoring_client.validate_auth()

    return scoring_client


def _setup_null_endpoint_scoring_client(
        configuration: Configuration,
        tally_handler: TallyFailedRequestHandler) -> ScoringClient:
    """Create an instance of a null endpoint scoring client."""
    scoring_client = NullEndpointScoringClient(
        http_response_handler=AoaiHttpResponseHandler(tally_handler, configuration),
        scoring_url=configuration.scoring_url)

    return scoring_client


def _setup_tally_handler(configuration):
    return TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions)
