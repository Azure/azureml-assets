# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LLM Client Class."""

from shared_utilities.llm_utils import (
    API_KEY,
    AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
    AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    _WorkspaceConnectionTokenManager,
    _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    get_llm_request_args
)
from shared_utilities.constants import (
    API_CALL_RETRY_BACKOFF_FACTOR,
    API_CALL_RETRY_MAX_COUNT
)


class LLMClient:
    """LLM Client Class."""

    def __init__(self,
                 workspace_connection_arm_id: str,
                 model_deployment_name: str) -> None:
        """Create a llm client.

        Args:
            workspace_connection_arm_id(str): azureml workspace connection arm id for llm.
            model_deployment_name(str): model deployment name of the connection.
        """
        self.workspace_connection_arm_id = workspace_connection_arm_id
        self.model_deployment_name = model_deployment_name
        self._setup_llm_properties()

    def _setup_llm_properties(self) -> None:  # noqa
        """Setup LLM related properties for later usage."""
        self.llm_request_args = get_llm_request_args(self.model_deployment_name)

        self.token_manager = _WorkspaceConnectionTokenManager(
            connection_name=self.workspace_connection_arm_id,
            auth_header=API_KEY)
        azure_endpoint_domain_name = self.token_manager.get_endpoint_domain().replace("https://", "")
        azure_openai_api_version = self.token_manager.get_api_version()

        self.azure_endpoint_url = _check_and_format_azure_endpoint_url(
            AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
            AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
            azure_endpoint_domain_name,
            azure_openai_api_version,
            self.model_deployment_name
        )

        self.http_client = _HTTPClientWithRetry(
            n_retry=API_CALL_RETRY_MAX_COUNT,
            backoff_factor=API_CALL_RETRY_BACKOFF_FACTOR,
        )
