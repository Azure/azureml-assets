# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Metrics violation index action detector class."""

from action_analyzer.contracts.action_detector import ActionDetector
from action_analyzer.contracts.action import Action
from typing import Dict
import pandas

SUPPORTED_METRICS = ["Fluency", "Coherence", "Relevance", "Groundedness", "RetrievalRelevance"]


class MetricsViolationIndexActionDetector(ActionDetector):
    """Metrics violation index action detector class."""

    def __init__(self,
                 workspace_connection_arm_id: str,
                 model_deployment_name: str,
                 aml_deployment_id: str,
                 metrics_violation_thresholds: Dict[str, float],
                 metrics_violation_passing_rate: Dict[str, float],
                 correlation_test_method: str,
                 correlation_test_pvalue_threshold: float,
                 action_max_positive_sample_size: int,
                 llm_summary_enabled: str) -> None:
        """Create a metrics violation index action detector.

        Args:
            workspace_connection_arm_id(str): azureml workspace connection arm id for llm.
            model_deployment_name(str): model deployment name of the connection.
            aml_deployment_id(str): the azureml deployment id of the llm application.
            metrics_violation_thresholds(Dict[str, float]): metrics violation thresholds dict, key is metric name, value is threshold for that metric.
            metrics_violation_passing_rate(Dict[str, float]): metrics violation passing rate dict, key is metric name, value is passing rate for that metric.
            correlation_test_method(str): test method for correlation test. e.g. ttest.
            correlation_test_pvalue_threshold(float): p-value threshold for correlation test to generate action.
            action_max_positive_sample_size(int): max number of positive samples in the action.
            llm_summary_enabled(str): enable llm summary. Accepted values: true or false.
        """
        self.metrics_violation_thresholds = metrics_violation_thresholds
        self.metrics_violation_passing_rate = metrics_violation_passing_rate
        self.correlation_test_method = correlation_test_method
        self.correlation_test_pvalue_threshold = correlation_test_pvalue_threshold
        super().__init__(workspace_connection_arm_id, model_deployment_name, aml_deployment_id, action_max_positive_sample_size, llm_summary_enabled)


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
            
            



    
