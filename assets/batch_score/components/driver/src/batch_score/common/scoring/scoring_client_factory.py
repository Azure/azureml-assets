# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for scoring client factory."""

import json

from ...aoai.scoring.aoai_scoring_client import AoaiScoringClient
from ...batch_pool.quota.quota_client import QuotaClient
from ...batch_pool.routing.routing_client import RoutingClient
from ...batch_pool.scoring.scoring_client import ScoringClient

from ...common.auth.auth_provider_factory import AuthProviderFactory
from ...common.auth.token_provider import TokenProvider
from ...common.common_enums import EndpointType
from ...common.configuration.configuration import Configuration
from ...common.configuration.metadata import Metadata
from ...common.header_providers.header_provider import HeaderProvider
from ...common.header_providers.header_provider_factory import HeaderProviderFactory
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.logging_utils import (
    get_logger,
    set_batch_pool,
    set_quota_audience,
)

from ...header_handlers.mir_and_batch_pool_header_handler_factory import MirAndBatchPoolHeaderHandlerFactory
from ...header_handlers.mir_endpoint_v2_header_handler import MIREndpointV2HeaderHandler

from ...mir.scoring.mir_scoring_client import MirScoringClient


class ScoringClientFactory:
    """Defines the scoring client factory."""

    def setup_scoring_client(
        self,
        configuration: Configuration,
        metadata: Metadata,
        token_provider: TokenProvider,
        routing_client: RoutingClient,
    ) -> "AoaiScoringClient | MirScoringClient | ScoringClient":
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

        if (configuration.is_aoai_endpoint() or configuration.is_serverless_endpoint()):
            get_logger().info("Using LLM scoring client.")
            return _setup_aoai_scoring_client(
                configuration=configuration,
                header_provider=header_provider,
                tally_handler=tally_handler)

        if configuration.get_endpoint_type() == EndpointType.BatchPool:
            get_logger().info("Using batch pool scoring client.")
            return _setup_pool_scoring_client(
                configuration=configuration,
                header_provider=header_provider,
                tally_handler=tally_handler,
                token_provider=token_provider,
                routing_client=routing_client)

        get_logger().info("Using MIR scoring client.")
        return _setup_mir_scoring_client(
            configuration=configuration,
            header_provider=header_provider,
            tally_handler=tally_handler)

    def _get_additional_headers(self, configuration):
        if configuration.additional_headers is None:
            headers = {}
        elif isinstance(configuration.additional_headers, dict):
            headers = configuration.additional_headers.copy()
        else:
            headers = json.loads(configuration.additional_headers)

        endpoint_type = configuration.get_endpoint_type()
        if endpoint_type == EndpointType.BatchPool or endpoint_type == EndpointType.MIR:
            headers = HeaderProviderFactory().set_defaults_for_openai_model_headers(
                headers=headers,
                configuration=configuration)

        return headers


def _setup_aoai_scoring_client(
        configuration: Configuration,
        header_provider: HeaderProvider,
        tally_handler: TallyFailedRequestHandler) -> AoaiScoringClient:
    """Create an instance of an AOAI scoring client."""
    scoring_client = AoaiScoringClient(
        header_provider=header_provider,
        scoring_url=configuration.scoring_url,
        tally_handler=tally_handler)

    scoring_client.validate_auth()

    return scoring_client


def _setup_mir_scoring_client(
        configuration: Configuration,
        header_provider: HeaderProvider,
        tally_handler: TallyFailedRequestHandler) -> ScoringClient:
    """Create an instance of an MIR scoring client.

    The client scores against a single MIR endpoint.
    """
    scoring_client = MirScoringClient(
        header_provider=header_provider,
        configuration=configuration,
        tally_handler=tally_handler)

    scoring_client.validate_auth()

    return scoring_client


def _setup_pool_scoring_client(
        configuration: Configuration,
        header_provider: HeaderProvider,
        tally_handler: TallyFailedRequestHandler,
        token_provider: TokenProvider,
        routing_client: RoutingClient) -> ScoringClient:
    """Create an instance of a batch pool scoring client.

    The client scores against a pool of MIR endpoints.
    """
    quota_client = _setup_quota_client(configuration, token_provider)

    return ScoringClient(
        header_provider=header_provider,
        quota_client=quota_client,
        routing_client=routing_client,
        scoring_url=configuration.scoring_url,
        tally_handler=tally_handler)


def _setup_header_handler(configuration, metadata, token_provider, connection_name):
    """Get the auth token from workspace connection and add that to the header.

    Additionally, add the 'Content-Type' header. The presence of workspace connection also
    signifies that this can be any MIR endpoint, so this will not add OAI specific headers.
    """
    if connection_name is not None:
        return MIREndpointV2HeaderHandler(connection_name, configuration.additional_headers)

    return MirAndBatchPoolHeaderHandlerFactory().get_header_handler(
        configuration=configuration,
        metadata=metadata,
        token_provider=token_provider)


def _setup_quota_client(configuration, token_provider):
    set_batch_pool(configuration.batch_pool)
    set_quota_audience(configuration.quota_audience)

    header_provider = HeaderProviderFactory().get_header_provider_for_rate_limiter(
        configuration=configuration,
        metadata=Metadata(),
        token_provider=token_provider)

    return QuotaClient(
        header_provider=header_provider,
        batch_pool=configuration.batch_pool,
        quota_audience=configuration.quota_audience,
        quota_estimator=configuration.quota_estimator,
        service_namespace=configuration.service_namespace)


def _setup_tally_handler(configuration):
    return TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions)
