# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metrics violation index action detector class."""

from action_analyzer.contracts.detectors.action_detector import ActionDetector
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.llm_client import LLMClient
from typing import Dict
import pandas

SUPPORTED_METRICS = ["Fluency", "Coherence", "Relevance", "Groundedness", "RetrievalRelevance"]


class MetricsViolationIndexActionDetector(ActionDetector):
    """Metrics violation index action detector class."""

    def __init__(self,
                 index_id: str,
                 violated_metrics: list[str],
                 correlation_test_method: str,
                 correlation_test_pvalue_threshold: float,
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str,
                 positive_metric_threshold = 5,
                 negative_metric_threshold = 3) -> None:
        """Create a metrics violation index action detector.

        Args:
            index_id(str): the index asset id.
            violated_metrics(List[str]): violated e2e metrics
            correlation_test_method(str): test method for correlation test. e.g. ttest.
            correlation_test_pvalue_threshold(float): p-value threshold for correlation test to generate action.
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm generated summary. Accepted values: true or false.
            positive_metrics_threshold(int): (Optional) e2e metric threshold to mark the query as positive. 
            negative_metrics_threshold(int): (Optional) e2e metric threshold to mark the query as negative.
        """

        self.correlation_test_method = correlation_test_method
        self.correlation_test_pvalue_threshold = correlation_test_pvalue_threshold
        self.positive_metrics_threshold = positive_metrics_threshold
        self.negative_metrics_threshold = negative_metrics_threshold
        super().__init__(action_max_positive_sample_size, llm_summary_enabled)


    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        pass


    def detect(self, df: pandas.DataFrame, llm_client: LLMClient) -> list(Action):
        """Detect the action.
        Args:
            df(pandas.DataFrame): input pandas dataframe.
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.

        Returns:
            list(Action): list of actions.
        """
        pass


    # def _get_violated_metrics(self, df) -> list[str]:
    #     """Get violated metrics.

    #     Args:
    #         df(pandas.DataFrame): input pandas dataframe.
    #     """
        
    #     column_names = df.columns.values.tolist()
    #     missing_metrics = []
    #     for metric, threshold in self.MetricsViolationThresholds.items():
    #         if metric.lower() not in [m.lower() for m in SUPPORTED_METRICS]:
    #             print(f"Metric {metric} is not supported. The supported metrics are: {SUPPORTED_METRICS}.")
    #         if metric.lower() not in [c.lower() for c in column_names]:
    #             print(f"Metric score {metrics} does not exist in the input dataframe.")
    #             missing_metrics.append(metric)
        
    #     if len(missing_metrics) == 0:
            
            



    
