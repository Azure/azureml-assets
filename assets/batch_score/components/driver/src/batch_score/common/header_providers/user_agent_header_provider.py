# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for user agent header provider."""

import os

from .header_provider import HeaderProvider
from ...common import constants


class UserAgentHeaderProvider(HeaderProvider):
    """User agent header provider."""

    def __init__(
            self,
            component_version: str,
            quota_audience: str = None,
            user_agent_segment: str = None):
        """Initialize the UserAgentHeaderProvider."""
        self._component_version = component_version
        self._quota_audience = quota_audience
        self._user_agent_segment = user_agent_segment

    def get_headers(self) -> dict:
        """Get the headers for requests."""
        return {'User-Agent': self._get_user_agent()}

    def _get_user_agent(self) -> str:
        """Get the user agent string."""
        if not self._quota_audience:
            return 'BatchScore:{}/Run:{}{}'.format(
                self._component_version,
                os.environ.get(constants.OS_ENVIRON_RUN_ID, "DNE"),
                f"/{self._user_agent_segment}" if self._user_agent_segment else ""
            )

        workload_id = ":".join(
            [x for x in [self._quota_audience, self._user_agent_segment] if x is not None])

        return 'BatchScore:{}/{}/Run:{}:{}'.format(
            self._component_version,
            workload_id if workload_id else "Unknown",
            os.environ.get(constants.OS_ENVIRON_WORKSPACE, "DNE"),
            os.environ.get(constants.OS_ENVIRON_RUN_ID, "DNE"))
