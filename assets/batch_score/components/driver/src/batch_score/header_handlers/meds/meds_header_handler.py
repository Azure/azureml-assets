# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MEDS header handler."""

from ...common.auth.token_provider import TokenProvider
from ..header_handler import HeaderHandler


class MedsHeaderHandler(HeaderHandler):
    """MEDS header handler."""

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get headers for MEDS requests."""
        bearer_token = self._token_provider.get_token(scope=TokenProvider.SCOPE_AML)
        user_agent = self._get_user_agent()

        headers = {
            'Authorization': 'Bearer ' + bearer_token,
            'User-Agent': user_agent,
        }

        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
