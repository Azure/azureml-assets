# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR and batch pool header handler factory."""

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
        token_provider: TokenProvider,
    ) -> OpenAIHeaderHandler:
        """Get the header provider for the given configuration."""
        if configuration.is_sahara():
            handler = SaharaHeaderHandler
        elif configuration.is_vesta():
            handler = VestaHeaderHandler
        elif configuration.is_vesta_chat_completion():
            handler = VestaChatCompletionHeaderHandler
        # TODO: Embeddings should probably have its own handler
        elif configuration.is_completion() or configuration.is_embeddings():
            handler = CompletionHeaderHandler
        elif configuration.is_chat_completion():
            handler = ChatCompletionHeaderHandler
        else:
            get_logger().info("No OpenAI model matched, defaulting to base OpenAI header handler.")
            handler = OpenAIHeaderHandler

        return handler(
            token_provider=token_provider,
            component_version=metadata.component_version,
            user_agent_segment=configuration.user_agent_segment,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            additional_headers=configuration.additional_headers
        )
