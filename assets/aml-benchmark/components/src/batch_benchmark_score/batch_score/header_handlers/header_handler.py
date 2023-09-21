# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The base class for header handlers.."""

import json
import os

from abc import ABC, abstractclassmethod
from azureml._restclient.clientbase import ClientBase
from azureml._model_management._util import get_requests_session
from ..utils.token_provider import TokenProvider
from ..utils.common import constants
from azureml.core import Workspace, Run
from azureml._common._error_definition.azureml_error import AzureMLError
from ..utils.exceptions import BenchmarkValidationException
from ..utils.error_definitions import BenchmarkValidationError


class HeaderHandler(ABC):
    """Class for header handler."""

    def __init__(self,
                 token_provider: TokenProvider,
                 user_agent_segment: str = None,
                 batch_pool: str = None,
                 quota_audience: str = None,
                 additional_headers: str = None) -> None:
        """Init method."""
        if user_agent_segment and (":" in user_agent_segment or "/" in user_agent_segment):
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError,
                    error_details="user_agent_segment should not contain characters ':' or '/'")
            )

        self._token_provider = token_provider
        self._user_agent_segment = user_agent_segment
        self._batch_pool = batch_pool
        self._quota_audience = quota_audience

        if additional_headers is not None:
            self._additional_headers = json.loads(additional_headers)
        else:
            self._additional_headers = {}

    @abstractclassmethod
    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get headers interface."""
        pass

    # read-only
    @property
    def batch_pool(self):
        """Batch pool."""
        return self._batch_pool

    # read-only
    @property
    def quota_audience(self):
        """Quota audience."""
        return self._quota_audience
    
    @property
    @abstractclassmethod
    def _auth_key_in_resp(self):
        pass

    @abstractclassmethod
    def _get_list_key_url(self, workspace: Workspace) -> str:
        pass

    def _get_user_agent(self) -> str:
        workload_id = ":".join(
            [x for x in [self._batch_pool, self._quota_audience, self._user_agent_segment] if x is not None])

        return 'BatchScoreComponent:{}/Workload:{}/Run:{}:{}'.format(
            constants.BATCH_SCORE_COMPONENT_VERSION,
            workload_id if workload_id else "Unknown",
            os.environ.get(constants.OS_ENVIRON_WORKSPACE, "DNE"),
            os.environ.get(constants.OS_ENVIRON_RUN_ID, "DNE"))
    
    def _get_auth_key(self):
        workspace = self._get_curr_workspace()
        key_endpoint = self._get_list_key_url(workspace)
        resp = ClientBase._execute_func(
            get_requests_session().post, key_endpoint, params={},
            headers=workspace._auth.get_authentication_header())
        return self._retrieve_key(resp)

    def _get_curr_workspace(self) -> Workspace:
        run = Run.get_context()
        curr_workspace = run.experiment.workspace
        return curr_workspace
    
    def _retrieve_key(self, resp) -> str:
        content = resp.content
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        keys_content = json.loads(content)
        return keys_content[self._auth_key_in_resp]
