# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""
from typing import Any, Optional, Dict

import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider

from aml_benchmark.utils.online_endpoint.aoai_online_endpoint import AOAIOnlineEndpoint
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel


class OAIHeaderHandler(HeaderHandler):
    """Class for OAI header handler."""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, endpoint_name: str = None,
            endpoint_subscription: str = None, endpoint_resource_group: str = None, deployment_name: str = None,
            connections_name: str = None, online_endpoint_model: OnlineEndpointModel = None
    ) -> None:
        """Init method."""
        super().__init__(
            token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        self._endpoint_name = endpoint_name
        self._endpoint_subscription = endpoint_subscription
        self._endpoint_resource_group = endpoint_resource_group
        self._deployment_name = deployment_name
        self._connections_name = connections_name
        self._model = online_endpoint_model

    def get_headers(self, additional_headers: Dict[str, Any] = None, palyoad: Optional[Any] = None) -> Dict[str, Any]:
        """Get handers."""
        online_endpoint = AOAIOnlineEndpoint(
            workspace_name=None, resource_group=self._endpoint_resource_group,
            subscription_id=self._endpoint_subscription, online_endpoint_model=self._model,
            endpoint_name=self._endpoint_name, deployment_name=self._deployment_name, sku=None,
            location=None, connections_name=self._connections_name
        )
        headers = online_endpoint.get_endpoint_authorization_header_from_connections()
        user_agent = self._get_user_agent()

        headers.update({
            'Content-Type': 'application/json',
            'User-Agent': user_agent,
            'x-ms-client-request-id': str(uuid.uuid4()),
        })
        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
