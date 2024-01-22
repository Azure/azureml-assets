# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for scoring client factory."""

from ...aoai.scoring.aoai_scoring_client import AoaiScoringClient
from ...batch_pool.quota.quota_client import QuotaClient
from ...batch_pool.routing.routing_client import RoutingClient
from ...batch_pool.scoring.scoring_client import ScoringClient

from ...common.auth.auth_provider_factory import AuthProviderFactory
from ...common.auth.token_provider import TokenProvider
from ...common.configuration.configuration import Configuration
from ...common.configuration.metadata import Metadata
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.logging_utils import (
    get_logger,
    set_batch_pool,
    set_quota_audience,
)

from ...header_handlers.mir_and_batch_pool_header_handler_factory import MirAndBatchPoolHeaderHandlerFactory
from ...header_handlers.mir_endpoint_v2_header_handler import MIREndpointV2HeaderHandler
from ...header_handlers.rate_limiter.rate_limiter_header_handler import (
    RateLimiterHeaderHandler,
)


class ScoringClientFactory:
    """Defines the scoring client factory."""

    def setup_scoring_client(
        self,
        configuration: Configuration,
        metadata: Metadata,
        token_provider: TokenProvider,
        routing_client: RoutingClient,
    ) -> "ScoringClient | AoaiScoringClient":
        """Gets the scoring client for the given configuration."""
        if (configuration.is_aoai_endpoint() or configuration.is_serverless_endpoint()):
            get_logger().info("Using LLM scoring client.")
            return setup_aoai_scoring_client(configuration=configuration)
        else:
            get_logger().info("Using MIR scoring client.")
            return setup_mir_scoring_client(
                configuration=configuration,
                metadata=metadata,
                token_provider=token_provider,
                routing_client=routing_client,
                connection_name=configuration.connection_name
            )


def setup_aoai_scoring_client(
        configuration: Configuration) -> AoaiScoringClient:
    tally_handler = TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions
    )
    """Creates an instance of an AOAI scoring client."""
    auth_provider = AuthProviderFactory().get_auth_provider(configuration)

    scoring_client = AoaiScoringClient(
        auth_provider=auth_provider,
        scoring_url=configuration.scoring_url,
        tally_handler=tally_handler,
        additional_headers=configuration.additional_headers
    )

    scoring_client.validate_auth()

    return scoring_client


def setup_mir_scoring_client(
        configuration: Configuration,
        metadata: Metadata,
        token_provider: TokenProvider,
        routing_client: RoutingClient,
        connection_name: str) -> ScoringClient:
    """Creates an instance of an MIR scoring client."""
    quota_client: QuotaClient = None
    if configuration.batch_pool and configuration.quota_audience and configuration.service_namespace:
        set_batch_pool(configuration.batch_pool)
        set_quota_audience(configuration.quota_audience)

        rate_limiter_header_handler = RateLimiterHeaderHandler(
            token_provider=token_provider,
            user_agent_segment=configuration.user_agent_segment,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience)

        quota_client = QuotaClient(
            header_handler=rate_limiter_header_handler,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            quota_estimator=configuration.quota_estimator,
            service_namespace=configuration.service_namespace
        )

    tally_handler = TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions
    )

    header_handler = None

    '''Get the auth token from workspace connection and add that to the header. Additionally, add the 'Content-Type' header.
    The presence of workspace connection also signifies that this can be any MIR endpoint, so this will not add OAI specific headers.
    '''    
    if connection_name is not None:
        header_handler = MIREndpointV2HeaderHandler(connection_name, configuration.additional_headers)
    else:
        header_handler = MirAndBatchPoolHeaderHandlerFactory().get_header_handler(
            configuration=configuration,
            metadata=metadata,
            routing_client=routing_client,
            token_provider=token_provider
        )

    scoring_client = ScoringClient(
        header_handler=header_handler,
        quota_client=quota_client,
        routing_client=routing_client,
        scoring_url=configuration.scoring_url,
        tally_handler=tally_handler,
    )

    return scoring_client
