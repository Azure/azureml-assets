# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Detector Class."""

from abc import ABC, abstractmethod
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient


class ActionDetector(ABC):
    """Action detector base class."""

    def __init__(self,
                 action_max_positive_sample_size: int,
                 query_intention_enabled: str) -> None:
        """Create an action detector.

        Args:
            action_max_positive_sample_size(int): max number of positive samples in the action.
            query_intention_enabled(str): enable llm generated query intention. Accepted values: true or false.
        """
        self.action_max_positive_sample_size = action_max_positive_sample_size
        self.query_intention_enabled = query_intention_enabled


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
    def detect(self, df: pandas.DataFrame, llm_client: LLMClient) -> list(Action):
        """Detect the action.
        Args:
            df(pandas.DataFrame): input pandas dataframe.
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.

        Returns:
            list(Action): list of actions.
        """
        pass
