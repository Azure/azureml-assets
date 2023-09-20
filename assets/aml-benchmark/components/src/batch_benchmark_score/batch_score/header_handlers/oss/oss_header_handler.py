# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for OSS header handler."""

import json
import uuid

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider
from ...utils.common import constants

from azureml.core import Run, Workspace
from azureml._restclient.clientbase import ClientBase
from azureml._model_management._util import get_requests_session
from azureml._model_management._util import _get_mms_url


class OSSHeaderHandler(HeaderHandler):
    """Class for OSS header handler"""

    def __init__(
            self,
            token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
            quota_audience: str = None, additional_headers: str = None, deployment_name: str = None,
            endpoint_subscription: str = None, endpoint_resource_group: str = None,
            endpoint_workspace: str = None
    ) -> None:
        """The init file."""
        super().__init__(token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        self._deployment_name = deployment_name
        self._endpoint_subscription = endpoint_subscription
        self._endpoint_resource_group = endpoint_resource_group
        self._endpoint_workspace = endpoint_workspace

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get handers."""
        bearer_token, _ = self._get_auth_key()

        print(f"bearer_token is {bearer_token}")

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

        print(headers)

        return headers

    def _get_auth_key(self):
        run = Run.get_context()
        curr_workspace = run.experiment.workspace
        if self._endpoint_workspace is None:
            workspace = curr_workspace
        else:
            workspace = Workspace(
                self._endpoint_subscription, self._endpoint_resource_group, self._endpoint_workspace,
                auth=curr_workspace._auth)
        headers = workspace._auth.get_authentication_header()
        list_keys_url = _get_mms_url(workspace) + '/onlineEndpoints/{}'.format(self._deployment_name) + '/listkeys'
        resp = ClientBase._execute_func(
            get_requests_session().post, list_keys_url, params={}, headers=headers)

        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        keys_content = json.loads(content)
        print(keys_content)
        primary_key = keys_content['primaryKey']
        secondary_key = keys_content['secondaryKey']
        return primary_key, secondary_key
