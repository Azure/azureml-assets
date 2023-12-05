# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for Claudie header handler."""
from typing import Any, Dict, Optional

import re

from aml_benchmark.batch_benchmark_score.batch_score.header_handlers.header_handler import HeaderHandler
from aml_benchmark.utils.online_endpoint.claude_online_endpoint import ClaudeOnlineEndpoint
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from aml_benchmark.batch_benchmark_score.batch_score.utils.token_provider import TokenProvider


class ClaudeHeaderHandler(HeaderHandler):
    """Class for ClaudeHeaderHandler."""

    def __init__(
        self,
        token_provider: TokenProvider, user_agent_segment: str = None, batch_pool: str = None,
        quota_audience: str = None, additional_headers: str = None, endpoint_name: str = None,
        endpoint_subscription: str = None, endpoint_resource_group: str = None, deployment_name: str = None,
        connections_name: str = None, online_endpoint_model: OnlineEndpointModel = None
    ) -> None:
        '''
        Constructor
        '''
        super().__init__(
            token_provider, user_agent_segment, batch_pool, quota_audience, additional_headers)
        match_regex = re.search(
            'bedrock-runtime.(?P<aws_region>.+).amazonaws.com/model/(?P<model_identifier>.+)/invoke',
            online_endpoint_model.model_path
        )
        self._aws_region = match_regex.group('aws_region')
        self._model_identifier = match_regex.group('model_identifier')
        self._model = online_endpoint_model
        self._connection_name = connections_name

    def get_headers(self, additional_headers: Dict[str, Any] = None, payload: Optional[Any] = None) -> Dict[str, Any]:
        """
        Return the authentication headers specific for the Claudie.

        :param additional_headers: The headers to be added to authentication headers.
        :param payload: the payload to be sent to the endpoint.
        :return: The headers for scoring on the endpoint.
        """

        endpoint = ClaudeOnlineEndpoint(
            workspace_name=None,
            resource_group=None,
            subscription_id=None,
            connections_name=self._connection_name,
            aws_region=self._aws_region,
            model_identifier=self._model_identifier,
            payload=payload,
            online_endpoint_model=self._model
        )
        headers = endpoint.get_endpoint_authorization_header()
        if additional_headers:
            headers.update(additional_headers)
        return headers
