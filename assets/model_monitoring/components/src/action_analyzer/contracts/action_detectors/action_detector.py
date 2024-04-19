# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Detector Class."""

from abc import ABC, abstractmethod
from action_analyzer.contracts.action import Action


class ActionDetector(ABC):
    """Action detector base class."""

    def __init__(self,
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str) -> None:
        """Create an action detector.

        Args:
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm generated summary. Accepted values: true or false.
        """
        self.action_max_positive_sample_size = action_max_positive_sample_size
        self.llm_summary_enabled = llm_summary_enabled


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
