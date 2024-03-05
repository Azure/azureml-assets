# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for auth header provider."""

from unittest.mock import MagicMock

import pytest

from src.batch_score.common.auth.auth_provider import WorkspaceConnectionAuthProvider, IdentityAuthProvider
from src.batch_score.common.configuration.configuration import EndpointType
from src.batch_score.common.header_providers.auth_header_provider import AuthHeaderProvider
from src.batch_score.common.header_providers.content_type_header_provider import ContentTypeHeaderProvider
from src.batch_score.common.header_providers.header_provider_factory import HeaderProviderFactory
from src.batch_score.common.header_providers.multi_header_provider import MultiHeaderProvider
from src.batch_score.common.header_providers.token_auth_header_provider import TokenAuthHeaderProvider
from src.batch_score.common.header_providers.traffic_group_header_provider import TrafficGroupHeaderProvider
from src.batch_score.common.header_providers.user_agent_header_provider import UserAgentHeaderProvider
from src.batch_score.common.header_providers.x_ms_client_request_id_header_provider import (
    XMsClientRequestIdHeaderProvider,
)


@pytest.mark.parametrize(
    "endpoint_type, auth_provider, expected_header_providers",
    [
        (
            EndpointType.AOAI,
            MagicMock(),
            [
                AuthHeaderProvider,
                ContentTypeHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
        ),
        (
            EndpointType.BatchPool,
            MagicMock(),
            [
                ContentTypeHeaderProvider,
                TokenAuthHeaderProvider,
                TrafficGroupHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
        ),
        (
            EndpointType.MIR,
            MagicMock(spec=IdentityAuthProvider),
            [
                ContentTypeHeaderProvider,
                TokenAuthHeaderProvider,
                TrafficGroupHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
        ),
        (
            EndpointType.MIR,
            MagicMock(spec=WorkspaceConnectionAuthProvider),
            [
                AuthHeaderProvider,
                ContentTypeHeaderProvider,
                TrafficGroupHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
        ),
        (
            EndpointType.Serverless,
            MagicMock(spec=WorkspaceConnectionAuthProvider),
            [
                AuthHeaderProvider,
                ContentTypeHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
        ),
    ],
)
def test_get_header_provider_for_scoring(
        endpoint_type,
        auth_provider,
        expected_header_providers):
    """Test get_header_provider_for_scoring."""
    configuration = MagicMock()
    configuration.get_endpoint_type.return_value = endpoint_type

    header_provider = HeaderProviderFactory().get_header_provider_for_scoring(
        auth_provider=auth_provider,
        configuration=configuration,
        metadata=MagicMock(),
        token_provider=MagicMock(),
    )

    assert isinstance(header_provider, MultiHeaderProvider)
    header_provider_types = [type(p) for p in header_provider._header_providers]
    assert set(header_provider_types) == set(expected_header_providers)


def test_get_header_provider_for_model_endpoint_discovery():
    """Test get_header_provider_for_model_endpoint_discovery."""
    header_provider = HeaderProviderFactory().get_header_provider_for_model_endpoint_discovery(
        configuration=MagicMock(),
        metadata=MagicMock(),
        token_provider=MagicMock(),
    )

    assert isinstance(header_provider, MultiHeaderProvider)
    header_provider_types = [type(p) for p in header_provider._header_providers]
    assert set(header_provider_types) == set([
        TokenAuthHeaderProvider,
        UserAgentHeaderProvider,
        XMsClientRequestIdHeaderProvider,
    ])


def test_get_header_provider_for_rate_limiter():
    """Test get_header_provider_for_rate_limiter."""
    header_provider = HeaderProviderFactory().get_header_provider_for_rate_limiter(
        configuration=MagicMock(),
        metadata=MagicMock(),
        token_provider=MagicMock(),
    )

    assert isinstance(header_provider, MultiHeaderProvider)
    header_provider_types = [type(p) for p in header_provider._header_providers]
    assert set(header_provider_types) == set([
        ContentTypeHeaderProvider,
        TokenAuthHeaderProvider,
        UserAgentHeaderProvider,
        XMsClientRequestIdHeaderProvider,
    ])


@pytest.mark.parametrize(
    "method_to_return_true, number_of_headers",
    [
        ("is_chat_completion", 4),
        ("is_completion", 0),
        ("is_embeddings", 0),
        ("is_sahara", 3),
        ("is_vesta", 0),
        ("is_vesta_chat_completion", 4),
        ("other_method", 0),
    ],
)
def test_set_defaults_for_openai_model_headers(method_to_return_true, number_of_headers):
    """Test set_defaults_for_openai_model_headers."""
    headers = {}
    configuration = MagicMock()

    def return_true(*args, **kwargs):
        return True

    def return_false(*args, **kwargs):
        return False

    methods_to_return_false = [
        'is_chat_completion',
        'is_completion',
        'is_embeddings',
        'is_sahara',
        'is_vesta',
        'is_vesta_chat_completion',
    ]
    for method in methods_to_return_false:
        setattr(configuration, method, return_false)

    setattr(configuration, method_to_return_true, return_true)
    headers_with_defaults = HeaderProviderFactory().set_defaults_for_openai_model_headers(
        headers=headers,
        configuration=configuration,
    )

    assert len(headers_with_defaults) == number_of_headers
