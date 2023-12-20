# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for Meds header handler."""
from typing import Any, Optional, Dict

from ..header_handler import HeaderHandler
from ...utils.token_provider import TokenProvider


class MedsHeaderHandler(HeaderHandler):
    """Class for MedsHeaderHandler."""

    def get_headers(self, additional_headers: Dict[str, Any] = None, payload: Optional[Any] = None) -> Dict[str, Any]:
        """Get header method."""
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
