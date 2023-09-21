# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""

import json
import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider
from ...utils.common import constants

from azureml.core import Workspace
from azureml._restclient.clientbase import ClientBase
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import _get_mms_url


class OSSHeaderHandler(HeaderHandler):
    """Class for OSS header handler."""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, deployment_name: str = None,
            endpoint_subscription: str = None, endpoint_resource_group: str = None,
            endpoint_workspace: str = None
    ) -> None:
        """Init method."""
        super().__init__(token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        self._deployment_name = deployment_name
        self._endpoint_subscription = endpoint_subscription
        self._endpoint_resource_group = endpoint_resource_group
        self._endpoint_workspace = endpoint_workspace

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get handers."""
        bearer_token = self._get_auth_key()
        user_agent = self._get_user_agent()

        headers = {
            'Authorization': f"Bearer {bearer_token}",
            'Content-Type': 'application/json',
            'User-Agent': user_agent,
            'azureml-model-group': constants.TRAFFIC_GROUP,
            'x-ms-client-request-id': str(uuid.uuid4()),
        }

        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
    
    def _get_list_key_url(self, workspace: Workspace) -> str:
        return _get_mms_url(workspace) + '/onlineEndpoints/{}'.format(self._deployment_name) + '/listkeys'

    @property
    def _auth_key_in_resp(self) -> str:
        return 'primaryKey'
    
    def _get_curr_workspace(self) -> Workspace:
        curr_workspace = super()._get_curr_workspace()
        if self._endpoint_workspace is None:
            workspace = curr_workspace
        else:
            workspace = Workspace(
                self._endpoint_subscription, self._endpoint_resource_group, self._endpoint_workspace,
                auth=curr_workspace._auth)
        return workspace
