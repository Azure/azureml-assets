# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Detector Class."""

from abc import ABC, abstractmethod
from shared_utilities.llm_utils import (
    API_KEY,
    AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
    AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
    _APITokenManager,
    _WorkspaceConnectionTokenManager,
    _HTTPClientWithRetry,
    _check_and_format_azure_endpoint_url,
    _request_api,
    get_llm_request_args
)
from shared_utilities.constants import (
    API_CALL_RETRY_BACKOFF_FACTOR,
    API_CALL_RETRY_MAX_COUNT
)
from action_analyzer.contracts.action import Action


class ActionDetector(ABC):
    """Action detector base class."""

    def __init__(self,
                 workspace_connection_arm_id: str,
                 model_deployment_name: str,
                 aml_deployment_id: str,
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str) -> None:
        """Create an action detector.

        Args:
            workspace_connection_arm_id(str): azureml workspace connection arm id for llm.
            model_deployment_name(str): model deployment name of the connection.
            aml_deployment_id(str): the azureml deployment id of the llm application.
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm summary. Accepted values: true or false.
        """
        self.aml_deployment_id = aml_deployment_id
        self.llm_summary_enabled = llm_summary_enabled
        self.action_max_positive_sample_size = action_max_positive_sample_size
        self.setup_llm_properties(workspace_connection_arm_id, model_deployment_name)

    def _setup_llm_properties(self, workspace_connection_arm_id: str, model_deployment_name: str) -> None:
        """Setup LLM related properties for later usage.

        Args:
            workspace_connection_arm_id(str): azureml workspace connection arm id for llm.
            model_deployment_name(str): model deployment name of the connection.
        """
        self.llm_request_args = get_llm_request_args(model_deployment_name)

        self.token_manager = _WorkspaceConnectionTokenManager(
            connection_name=workspace_connection_arm_id,
            auth_header=API_KEY)
        azure_endpoint_domain_name = token_manager.get_endpoint_domain().replace("https://", "")
        azure_openai_api_version = token_manager.get_api_version()

        self.azure_endpoint_url = _check_and_format_azure_endpoint_url(
            AZURE_OPENAI_API_COMPLETION_URL_PATTERN,
            AZURE_ENDPOINT_DOMAIN_VALID_PATTERN_RE,
            azure_endpoint_domain_name,
            azure_openai_api_version,
            model_deployment_name
        )

        self.http_client = _HTTPClientWithRetry(
            n_retry=API_CALL_RETRY_MAX_COUNT,
            backoff_factor=API_CALL_RETRY_BACKOFF_FACTOR,
        )


    @abstractmethod
    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        pass


    @abstractmethod
    def detect(self, df) -> list(Action):
        """Detect the action.
        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            list(Action): list of actions.
        """
        pass
