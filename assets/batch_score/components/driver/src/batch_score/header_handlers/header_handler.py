import json
import os
from abc import ABC, abstractclassmethod

from ..common import constants
from ..common.auth.token_provider import TokenProvider


class HeaderHandler(ABC):
    def __init__(
            self,
            token_provider: TokenProvider,
            component_version: str = None,
            user_agent_segment: str = None,
            batch_pool: str = None,
            quota_audience: str = None,
            additional_headers: str = None) -> None:
        if user_agent_segment and (":" in user_agent_segment or "/" in user_agent_segment):
            raise Exception("user_agent_segment should not contain characters ':' or '/'")

        self._token_provider = token_provider
        self._component_version = component_version
        self._user_agent_segment = user_agent_segment
        self._batch_pool = batch_pool
        self._quota_audience = quota_audience

        if additional_headers is not None:
            self._additional_headers = json.loads(additional_headers)
        else:
            self._additional_headers = {}

    @abstractclassmethod
    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        pass

    # read-only
    @property
    def batch_pool(self):
        return self._batch_pool

    # read-only
    @property
    def quota_audience(self):
        return self._quota_audience

    def _get_user_agent(self) -> str:
        workload_id = ":".join(
            [x for x in [self._batch_pool, self._quota_audience, self._user_agent_segment] if x is not None])

        return 'BatchScore:{}/{}/Run:{}:{}'.format(
            self._component_version,
            workload_id if workload_id else "Unknown",
            os.environ.get(constants.OS_ENVIRON_WORKSPACE, "DNE"),
            os.environ.get(constants.OS_ENVIRON_RUN_ID, "DNE"))
