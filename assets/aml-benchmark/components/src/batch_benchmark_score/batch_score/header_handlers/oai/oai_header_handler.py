# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""

import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider

from azureml.core import Workspace


class OAIHeaderHandler(HeaderHandler):
    """Class for OSS header handler."""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, deployment_name: str = None,
            use_managed_identity: bool = False
    ) -> None:
        """Init method."""
        super().__init__(
            token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers, use_managed_identity)
        self._deployment_name = deployment_name

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get handers."""
        token = self._get_auth_key()
        user_agent = self._get_user_agent()

        headers = {
            'api-key': token,
            'Content-Type': 'application/json',
            'User-Agent': user_agent,
            'x-ms-client-request-id': str(uuid.uuid4()),
        }
        print(headers)
        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
    
    @property
    def _auth_key_in_resp(self) -> str:
        return 'key1'

    def _get_list_key_url(self, workspace: Workspace) -> str:
        url_list = [
            'https://management.azure.com', 'subscriptions', workspace.subscription_id,
            'resourceGroups', workspace.resource_group, 'providers/Microsoft.CognitiveServices/accounts',
            self._deployment_name, 'listKeys?api-version=2023-05-01'
        ]
        return '/'.join(url_list)
