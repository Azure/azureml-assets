# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""OpenAI header handler."""

import uuid

from ...common import constants
from ...common.auth.token_provider import TokenProvider
from ..header_handler import HeaderHandler


class OpenAIHeaderHandler(HeaderHandler):
    """OpenAI header handler."""

    def __init__(self,
                 token_provider: TokenProvider,
                 user_agent_segment: str = None,
                 batch_pool: str = None,
                 quota_audience: str = None,
                 additional_headers: str = None) -> None:
        """Init function."""
        super().__init__(token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        if batch_pool is not None:
            self.__set_default_additional_headers()

    def get_headers(self, additional_headers: "dict[str, any]" = None) -> "dict[str, any]":
        """Get headers for OpenAI requests."""
        bearer_token = self._token_provider.get_token(scope=TokenProvider.SCOPE_AML)
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

    def __set_default_additional_headers(self):
        """Set the default values for case-insensitive keys."""
        azureml_collect_request_key = "azureml-collect-request"
        azureml_inferencing_offer_name_key = "azureml-inferencing-offer-name"

        if not self.__is_case_insensitive_key_in_additional_header(azureml_collect_request_key):
            self._additional_headers[azureml_collect_request_key] = 'false'

        if not self.__is_case_insensitive_key_in_additional_header(azureml_inferencing_offer_name_key):
            self._additional_headers[azureml_inferencing_offer_name_key] = 'azureml_vanilla'

    def __is_case_insensitive_key_in_additional_header(self, key_to_find):
        for key in self._additional_headers.keys():
            if key.lower() == key_to_find:
                return True
        return False
