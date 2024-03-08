# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for header provider factory."""

from .auth_header_provider import AuthHeaderProvider
from .content_type_header_provider import ContentTypeHeaderProvider
from .header_provider import HeaderProvider
from .multi_header_provider import MultiHeaderProvider
from .token_auth_header_provider import TokenAuthHeaderProvider
from .traffic_group_header_provider import TrafficGroupHeaderProvider
from .user_agent_header_provider import UserAgentHeaderProvider
from .x_ms_client_request_id_header_provider import XMsClientRequestIdHeaderProvider
from ..auth.auth_provider import AuthProvider, WorkspaceConnectionAuthProvider
from ..auth.token_provider import TokenProvider
from ..configuration.configuration import Configuration, EndpointType
from ..configuration.metadata import Metadata


class HeaderProviderFactory:
    """Header provider factory."""

    def get_header_provider_for_scoring(
            self,
            auth_provider: AuthProvider,
            configuration: Configuration,
            metadata: Metadata,
            token_provider: TokenProvider,
            additional_headers: dict = None) -> HeaderProvider:
        """Get an instance of header provider."""
        header_providers = [
            ContentTypeHeaderProvider(),
            UserAgentHeaderProvider(
                component_version=metadata.component_version,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                user_agent_segment=configuration.user_agent_segment),
            XMsClientRequestIdHeaderProvider(),
        ]

        if configuration.get_endpoint_type() in [EndpointType.AOAI, EndpointType.Serverless]:
            header_providers.append(AuthHeaderProvider(auth_provider))
        elif configuration.get_endpoint_type() == EndpointType.MIR:
            header_providers.append(TrafficGroupHeaderProvider())
            if isinstance(auth_provider, WorkspaceConnectionAuthProvider):
                header_providers.append(AuthHeaderProvider(auth_provider))
            else:
                header_providers.append(TokenAuthHeaderProvider(
                    token_provider=token_provider,
                    token_scope=TokenProvider.SCOPE_AML))
        else:
            header_providers.append(TrafficGroupHeaderProvider())
            header_providers.append(TokenAuthHeaderProvider(
                token_provider=token_provider,
                token_scope=TokenProvider.SCOPE_AML))

        return MultiHeaderProvider(
            header_providers=header_providers,
            additional_headers=additional_headers)

    def get_header_provider_for_model_endpoint_discovery(
            self,
            configuration: Configuration,
            metadata: Metadata,
            token_provider: TokenProvider,
            additional_headers: dict = None) -> HeaderProvider:
        """Get headers for MEDS requests."""
        header_providers = [
            UserAgentHeaderProvider(
                component_version=metadata.component_version,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                user_agent_segment=configuration.user_agent_segment),
            XMsClientRequestIdHeaderProvider(),
            TokenAuthHeaderProvider(
                token_provider=token_provider,
                token_scope=TokenProvider.SCOPE_AML),
        ]

        return MultiHeaderProvider(
            header_providers=header_providers,
            additional_headers=additional_headers)

    def get_header_provider_for_rate_limiter(
            self,
            configuration: Configuration,
            metadata: Metadata,
            token_provider: TokenProvider,
            additional_headers: dict = None) -> HeaderProvider:
        """Get header provider for rate limiter requests."""
        header_providers = [
            ContentTypeHeaderProvider(),
            UserAgentHeaderProvider(
                component_version=metadata.component_version,
                batch_pool=configuration.batch_pool,
                quota_audience=configuration.quota_audience,
                user_agent_segment=configuration.user_agent_segment),
            XMsClientRequestIdHeaderProvider(),
            TokenAuthHeaderProvider(
                token_provider=token_provider,
                token_scope=TokenProvider.SCOPE_ARM),
        ]

        return MultiHeaderProvider(
            header_providers=header_providers,
            additional_headers=additional_headers)

    def set_defaults_for_openai_model_headers(self, headers: dict, configuration: Configuration) -> dict:
        """Set default headers for OpenAI model requests."""
        headers = headers.copy()

        if configuration.is_chat_completion():
            key = 'chat_completion'
        elif configuration.is_completion():
            key = 'completion'
        elif configuration.is_embeddings():
            key = 'embedding'
        elif configuration.is_sahara():
            key = 'sahara'
        elif configuration.is_vesta():
            key = 'vesta'
        elif configuration.is_vesta_chat_completion():
            key = 'vesta_chat_completion'
        else:
            return headers

        for header_key, header_value in OPENAI_MODEL_HEADER_DEFAULTS[key].items():
            headers.setdefault(header_key, header_value)

        return headers


OPENAI_MODEL_HEADER_DEFAULTS = {
    "chat_completion": {
        "Openai-Internal-AllowChatCompletion": "true",
        "Openai-Internal-AllowedOutputSpecialTokens": "<|im_start|>,<|im_sep|>,<|im_end|>",
        "Openai-Internal-AllowedSpecialTokens": "<|im_start|>,<|im_sep|>,<|im_end|>",
        "Openai-Internal-HarmonyVersion": "harmony_v3",
    },
    "completion": {},
    "embedding": {},
    "sahara": {
        "Openai-Internal-AllowedOutputSpecialTokens": "<|im_start|>,<|im_sep|>,<|im_end|>",
        "Openai-Internal-AllowedSpecialTokens": "<|im_start|>,<|im_sep|>,<|im_end|>",
        "Openai-Internal-HarmonyVersion": "harmony_v3",
    },
    "vesta": {},
    "vesta_chat_completion": {
        "Openai-Internal-AllowedOutputSpecialTokens":
            "<|im_start|>,<|im_sep|>,<|im_end|>,<|diff_marker|>,<|fim_suffix|>,"
            "<|ghreview|>,<|ipynb_marker|>,<|fim_middle|>,<|meta_start|>",
        "Openai-Internal-AllowedSpecialTokens":
            "<|im_start|>,<|im_sep|>,<|im_end|>,<|diff_marker|>,<|fim_suffix|>,"
            "<|ghreview|>,<|ipynb_marker|>,<|fim_middle|>,<|meta_start|>",
        "Openai-Internal-HarmonyVersion": "harmony_v4.0.11_8k_turbo_mm",
        "Openai-Internal-MegatokenSwitchAndParams": "true-true-1-4-2000",
    },
}
