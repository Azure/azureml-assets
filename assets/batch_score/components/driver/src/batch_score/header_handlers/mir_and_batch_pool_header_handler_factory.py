# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR and batch pool header handler factory."""

from ..batch_pool.routing.routing_client import RoutingClient
from ..common.auth.token_provider import TokenProvider
from ..common.configuration.configuration import Configuration
from ..common.configuration.metadata import Metadata
from ..common.telemetry.logging_utils import get_logger
from .open_ai import (
    ChatCompletionHeaderHandler,
    CompletionHeaderHandler,
    OpenAIHeaderHandler,
    SaharaHeaderHandler,
    VestaHeaderHandler,
)
from .open_ai.vesta_chat_completion_header_handler import VestaChatCompletionHeaderHandler


class MirAndBatchPoolHeaderHandlerFactory:
    """Defines the header provider factory for MIR and batch pool."""

    def get_header_handler(
        self,
        configuration: Configuration,
        metadata: Metadata,
        routing_client: RoutingClient,
        token_provider: TokenProvider,
    ) -> OpenAIHeaderHandler:
        """Get the header provider for the given configuration."""
        if configuration.is_sahara(routing_client=routing_client):
            return SaharaHeaderHandler(
                token_provider=token_provider,
                component_version=metadata.component_version,
                user_agent_segment=configuration.user_agent_segment,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                additional_headers=configuration.additional_headers
            )
        if configuration.is_vesta():
            return VestaHeaderHandler(
                token_provider=token_provider,
                component_version=metadata.component_version,
                user_agent_segment=configuration.user_agent_segment,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                additional_headers=configuration.additional_headers
            )
        if configuration.is_vesta_chat_completion():
            return VestaChatCompletionHeaderHandler(
                token_provider=token_provider,
                component_version=metadata.component_version,
                user_agent_segment=configuration.user_agent_segment,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                additional_headers=configuration.additional_headers
            )
        # TODO: Embeddings should probably have its own handler
        if configuration.is_completion() or configuration.is_embeddings():
            return CompletionHeaderHandler(
                token_provider=token_provider,
                component_version=metadata.component_version,
                user_agent_segment=configuration.user_agent_segment,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                additional_headers=configuration.additional_headers
            )
        if configuration.is_chat_completion():
            return ChatCompletionHeaderHandler(
                token_provider=token_provider,
                component_version=metadata.component_version,
                user_agent_segment=configuration.user_agent_segment,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                additional_headers=configuration.additional_headers
            )

        get_logger().info("No OpenAI model matched, defaulting to base OpenAI header handler.")
        return OpenAIHeaderHandler(
            token_provider=token_provider,
            component_version=metadata.component_version,
            user_agent_segment=configuration.user_agent_segment,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            additional_headers=configuration.additional_headers
        )
