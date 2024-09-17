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
from ..auth.auth_provider import AuthProvider
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
        endpoint_type = configuration.get_endpoint_type()
        if not configuration.user_agent_segment:
            header_providers = [
                ContentTypeHeaderProvider(),
                XMsClientRequestIdHeaderProvider(),
            ]
        else:
            header_providers = [
                ContentTypeHeaderProvider(),
                UserAgentHeaderProvider(
                    component_version=metadata.component_version,
                    user_agent_segment=configuration.user_agent_segment),
                XMsClientRequestIdHeaderProvider(),
            ]

        if endpoint_type in [EndpointType.Serverless]:
            header_providers.append(AuthHeaderProvider(auth_provider))
        else:
            header_providers.append(TrafficGroupHeaderProvider())
            header_providers.append(TokenAuthHeaderProvider(
                token_provider=token_provider,
                token_scope=TokenProvider.SCOPE_AML))

        return MultiHeaderProvider(
            header_providers=header_providers,
            additional_headers=additional_headers)
