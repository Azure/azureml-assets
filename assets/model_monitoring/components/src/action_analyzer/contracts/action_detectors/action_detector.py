# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Detector Class."""

import pandas
from abc import ABC, abstractmethod
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient
from shared_utilities.constants import MAX_SAMPLE_SIZE

class ActionDetector(ABC):
    """Action detector base class."""

    def __init__(self,
                 query_intention_enabled: str,
                 max_positive_sample_size=MAX_SAMPLE_SIZE) -> None:
        """Create an action detector.

        Args:
            query_intention_enabled(str): enable llm generated query intention. Accepted values: true or false.
            max_positive_sample_size(int): (Optional) max positive sample size in the action.
        """
        self.query_intention_enabled = query_intention_enabled
        self.max_positive_sample_size = max_positive_sample_size

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
    def detect(self, df: pandas.DataFrame, llm_client: LLMClient, aml_deployment_id=None) -> list[Action]:
        """Detect the action.

        Args:
            df(pandas.DataFrame): input pandas dataframe.
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.
            aml_deployment_id(str): (Optional) aml deployment id for the action.

        Returns:
            list[Action]: list of actions.
        """
        pass
