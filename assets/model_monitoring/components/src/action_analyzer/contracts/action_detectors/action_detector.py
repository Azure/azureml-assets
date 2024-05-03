# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Detector Class."""

import pandas
from typing import List
from abc import ABC, abstractmethod
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient


class ActionDetector(ABC):
    """Action detector base class."""

    def __init__(self,
                 query_intention_enabled: str,
                 preprocessed_data: pandas.DataFrame=pandas.DataFrame()) -> None:
        """Create an action detector.

        Args:
            query_intention_enabled(str): enable llm generated query intention. Accepted values: true or false.
            preprocessed_data(pandas.DataFrame): (Optional) preprocessed data. If passed, skip the preprocess step.
        """
        self.query_intention_enabled = query_intention_enabled
        self.preprocessed_data = preprocessed_data

    @abstractmethod
    def preprocess_data(self, df: pandas.DataFrame):
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.
        """
        pass

    @abstractmethod
    def detect(self, llm_client: LLMClient, aml_deployment_id=None) -> List[Action]:
        """Detect the action.

        Args:
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.
            aml_deployment_id(str): (Optional) aml deployment id for the action.

        Returns:
            List[Action]: list of actions.
        """
        pass
