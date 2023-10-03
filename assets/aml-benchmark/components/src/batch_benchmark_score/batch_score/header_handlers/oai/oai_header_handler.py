# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""

import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider

from azureml.core import Workspace
from utils.online_endpoint.aoai_online_endpoint import AOAIOnlineEndpoint


class OAIHeaderHandler(HeaderHandler):
    """Class for OSS header handler."""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, endpoint_name: str = None,
            endpoint_subscription: str = None, endpoint_resource_group: str = None, deployment_name: str = None
    ) -> None:
        """Init method."""
        super().__init__(
            token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        self._endpoint_name = endpoint_name
        self._endpoint_subscription = endpoint_subscription
        self._endpoint_resource_group = endpoint_resource_group
        self._deployment_name = deployment_name

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get handers."""
        online_endpoint = AOAIOnlineEndpoint(
            workspace_name=None, resource_group=self._endpoint_resource_group,
            subscription_id=self._endpoint_subscription, online_endpoint_model=None,
            endpoint_name=self._endpoint_name, deployment_name=self._deployment_name, sku=None, location=None
        )
        headers = online_endpoint.get_endpoint_authorization_header()
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
    