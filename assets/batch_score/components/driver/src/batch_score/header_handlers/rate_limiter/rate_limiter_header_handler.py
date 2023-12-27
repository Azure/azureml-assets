# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Rate limiter header handler."""

from ...common.auth.token_provider import TokenProvider
from ..header_handler import HeaderHandler


class RateLimiterHeaderHandler(HeaderHandler):
    """Rate limiter header handler."""

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get headers for rate limiter requests."""
        bearer_token = self._token_provider.get_token(scope=TokenProvider.SCOPE_ARM)
        user_agent = self._get_user_agent()

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }

        headers.update(self._additional_headers)

        if additional_headers:
            headers.update(additional_headers)

        return headers
