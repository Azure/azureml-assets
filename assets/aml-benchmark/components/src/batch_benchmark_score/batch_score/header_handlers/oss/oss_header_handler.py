# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""

import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider
from ...utils.common import constants

from utils.online_endpoint.oss_online_endpoint import OSSOnlineEndpoint


class OSSHeaderHandler(HeaderHandler):
    """Class for OSS header handler."""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, endpoint_name: str = None,
            endpoint_subscription: str = None, endpoint_resource_group: str = None,
            endpoint_workspace: str = None
    ) -> None:
        """Init method."""
        super().__init__(token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        self._endpoint_name = endpoint_name
        self._endpoint_subscription = endpoint_subscription
        self._endpoint_resource_group = endpoint_resource_group
        self._endpoint_workspace = endpoint_workspace

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get handers."""
        online_endpoint = OSSOnlineEndpoint(
            workspace_name=self._endpoint_workspace, resource_group=self._endpoint_resource_group,
            subscription_id=self._endpoint_subscription, online_endpoint_model=None,
            endpoint_name=self._endpoint_name, deployment_name=None, sku=None
        )
        headers = online_endpoint.get_endpoint_authorization_header()
        user_agent = self._get_user_agent()

        headers.update({
            'Content-Type': 'application/json',
            'User-Agent': user_agent,
            'azureml-model-group': constants.TRAFFIC_GROUP,
            'x-ms-client-request-id': str(uuid.uuid4()),
        })

        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
