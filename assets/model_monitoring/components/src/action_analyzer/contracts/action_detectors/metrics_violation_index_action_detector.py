# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metrics violation index action detector class."""

from typing import List
from action_analyzer.contracts.action_detectors.action_detector import ActionDetector
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient
import pandas
from action_analyzer.contracts.utils.detector_utils import (
    extract_fields_from_debugging_info
)


class MetricsViolationIndexActionDetector(ActionDetector):
    """Metrics violation index action detector class."""

    def __init__(self,
                 index_id: str,
                 violated_metrics: List[str],
                 correlation_test_method: str,
                 correlation_test_pvalue_threshold: float,
                 query_intention_enabled: str,
                 positive_metric_threshold=5,
                 negative_metric_threshold=3) -> None:
        """Create a metrics violation index action detector.

        Args:
            index_id(str): the hashed index id.
            violated_metrics(List[str]): violated e2e metrics
            correlation_test_method(str): test method for correlation test. e.g. ttest.
            correlation_test_pvalue_threshold(float): p-value threshold for correlation test to generate action.
            query_intention_enabled(str): enable llm generated summary. Accepted values: true or false.
            positive_metric_threshold(int): (Optional) e2e metric threshold to mark the query as positive.
            negative_metric_threshold(int): (Optional) e2e metric threshold to mark the query as negative.
            max_positive_sample_size(int): (Optional) max positive sample size in the action.
        """
        self.correlation_test_method = correlation_test_method
        self.correlation_test_pvalue_threshold = correlation_test_pvalue_threshold
        self.positive_metric_threshold = positive_metric_threshold
        self.negative_metric_threshold = negative_metric_threshold
        super().__init__(query_intention_enabled)

    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        try:
            preprocessed_df = extract_fields_from_debugging_info(df, self.index_id)
            return preprocessed_df
        except Exception as e:
            print("MetricsViolationIndexActionDetector preprocess failed with error", e)
            return pandas.DataFrame()

    def detect(self, df: pandas.DataFrame, llm_client: LLMClient) -> List[Action]:
        """Detect the action.

        Args:
            df(pandas.DataFrame): input pandas dataframe.
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.

        Returns:
            List[Action]: list of actions.
        """
        pass
