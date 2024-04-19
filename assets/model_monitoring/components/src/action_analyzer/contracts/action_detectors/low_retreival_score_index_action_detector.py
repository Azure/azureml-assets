# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Low retrieval score index action detector class."""

from action_analyzer.contracts.action_detector import ActionDetector
from action_analyzer.contracts.action import Action


class LowRetreivalScoreIndexActionDetector(ActionDetector):
    """Low retrieval score index action detector class."""

    def __init__(self, 
                 index_id: str,
                 violated_metrics: list[str],
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str):
        """Create a low retrieval score index action detector.

        Args:
            index_id(str): the index asset id.
            violated_metrics(List[str]): violated e2e metrics
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm generated summary. Accepted values: true or false.
        """
        self.index_id = index_id
        self.violated_metrics = violated_metrics
        super().__init__(action_max_positive_sample_size, llm_summary_enabled)


    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        pass


    def detect(self, df) -> list(Action):
        """Detect the action.
        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            list(Action): list of actions.
        """
        pass
