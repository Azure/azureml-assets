# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for auth header provider."""

from unittest.mock import MagicMock

import pytest

from src.batch_score_oss.common.auth.auth_provider import WorkspaceConnectionAuthProvider
from src.batch_score_oss.common.configuration.configuration import EndpointType
from src.batch_score_oss.common.header_providers.auth_header_provider import AuthHeaderProvider
from src.batch_score_oss.common.header_providers.content_type_header_provider import ContentTypeHeaderProvider
from src.batch_score_oss.common.header_providers.header_provider_factory import HeaderProviderFactory
from src.batch_score_oss.common.header_providers.multi_header_provider import MultiHeaderProvider
from src.batch_score_oss.common.header_providers.user_agent_header_provider import UserAgentHeaderProvider
from src.batch_score_oss.common.header_providers.x_ms_client_request_id_header_provider import (
    XMsClientRequestIdHeaderProvider,
)


@pytest.mark.parametrize(
    "endpoint_type, auth_provider, expected_header_providers, user_agent_segment",
    [
        (
            EndpointType.Serverless,
            MagicMock(spec=WorkspaceConnectionAuthProvider),
            [
                AuthHeaderProvider,
                ContentTypeHeaderProvider,
                UserAgentHeaderProvider,
                XMsClientRequestIdHeaderProvider,
            ],
            "UA_segment"
        ),
    ],
)
def test_get_header_provider_for_scoring(
        endpoint_type,
        auth_provider,
        expected_header_providers,
        user_agent_segment):
    """Test get_header_provider_for_scoring."""
    configuration = MagicMock()
    configuration.get_endpoint_type.return_value = endpoint_type
    configuration.user_agent_segment = user_agent_segment

    header_provider = HeaderProviderFactory().get_header_provider_for_scoring(
        auth_provider=auth_provider,
        configuration=configuration,
        metadata=MagicMock(),
        token_provider=MagicMock(),
    )

    assert isinstance(header_provider, MultiHeaderProvider)
    header_provider_types = [type(p) for p in header_provider._header_providers]
    assert set(header_provider_types) == set(expected_header_providers)
