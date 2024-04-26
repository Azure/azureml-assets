# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Low retrieval score index action detector class."""

import os
import pandas
from action_analyzer.contracts.action_detectors.action_detector import ActionDetector
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.actions.low_retreival_score_index_action_detector import LowRetreivalScoreIndexAction
from action_analyzer.contracts.llm_client import LLMClient
from action_analyzer.contracts.utils import (
    extract_fields_from_debugging_info,
    get_missed_metrics,
    calculate_e2e_metrics,
    get_retrieval_score,
    get_query_intention,
    generate_index_action_samples
)
from shared_utilities.constants import (
    INDEX_SCORE_LLM_COLUMN,
    METRICS_VIOLATION_THRESHOLD,
    GOOD_METRICS_THRESHOLD,
    MAX_SAMPLE_SIZE,
    DEFAULT_TOPIC_NAME,
    INDEX_CONTENT_COLUMN,
    PROMPT_COLUMN,
    DEFAULT_LLM_SCORE
)


LOW_RETRIEVAL_SCORE_QUERY_RATIO_THRESHOLD = 0.1


class LowRetreivalScoreIndexActionDetector(ActionDetector):
    """Low retrieval score index action detector class."""

    def __init__(self,
                 index_id: str,
                 violated_metrics: list[str],
                 llm_summary_enabled: str,
                 max_positive_sample_size=MAX_SAMPLE_SIZE) -> None:
        """Create a low retrieval score index action detector.

        Args:
            index_id(str): the index asset id.
            violated_metrics(List[str]): violated e2e metrics
            llm_summary_enabled(str): enable llm generated summary. Accepted values: true or false.
            max_positive_sample_size(int): (Optional) max positive sample size in the action.
        """
        self.index_id = index_id
        self.violated_metrics = violated_metrics
        super().__init__(llm_summary_enabled, max_positive_sample_size)

    def preprocess_data(self, df: pandas.DataFrame) -> pandas.DataFrame:
        """Preprocess the data for action detector.

            1. check if all violated metrics are available. If not, call evaluate sdk to get the e2e metrics.
            2. extract extra fields from the root span for action.
            3. convert the dataframe from trace level to span level.

        Args:
            df(pandas.DataFrame): input pandas dataframe.

        Returns:
            pandas.DataFrame: preprocessed pandas dataframe.
        """
        missed_metrics = get_missed_metrics(self.violated_metrics, df.columns.tolist())

        if missed_metrics != []:
            df = calculate_e2e_metrics(df, missed_metrics)

        return extract_fields_from_debugging_info(df, self.index_id)

    def detect(self, df: pandas.DataFrame, llm_client: LLMClient, aml_deployment_id=None) -> list[Action]:
        """Detect the action.

        Args:
            df(pandas.DataFrame): input pandas dataframe.
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.
            aml_deployment_id(str): (Optional) aml deployment id for the action.

        Returns:
            list[Action]: list of actions.
        """
        # get llm retrieval score
        df[INDEX_SCORE_LLM_COLUMN] = df.apply(get_retrieval_score, axis=1, args=(llm_client,))
        df = df[df[INDEX_SCORE_LLM_COLUMN] != DEFAULT_LLM_SCORE]

        action_list = []
        for metric in self.violated_metrics:
            low_retrieval_score_df = df[df[metric] < METRICS_VIOLATION_THRESHOLD &
                                        df[INDEX_SCORE_LLM_COLUMN] < METRICS_VIOLATION_THRESHOLD]
            high_retrieval_score_df = df[df[metric] >= GOOD_METRICS_THRESHOLD &
                                         df[INDEX_SCORE_LLM_COLUMN] >= GOOD_METRICS_THRESHOLD]
            # generate action only low retrieval score query ratio above the threshold
            low_retrieval_score_query_ratio = len(low_retrieval_score_df)/len(df)
            if low_retrieval_score_query_ratio >= LOW_RETRIEVAL_SCORE_QUERY_RATIO_THRESHOLD:
                print(f"Generating action for metric {metric}. \
                The low retrieval score query ratio is {low_retrieval_score_query_ratio}.")
                # use the low retrieval score query ratio as confidence
                action = self.generate_action(metric,
                                              low_retrieval_score_query_ratio,
                                              low_retrieval_score_df,
                                              high_retrieval_score_df,
                                              aml_deployment_id)
                action_list.append(action)
        return action_list

    def generate_action(self,
                        metric: str,
                        confidence_score: float,
                        low_retrieval_score_df: pandas.DataFrame,
                        high_retrieval_score_df: pandas.DataFrame,
                        aml_deployment_id: str) -> LowRetreivalScoreIndexAction:
        """Generate action from the dataframe.

        Args:
            metric(str): the violated metric for action.
            confidence_score(float): LLM client used to get some llm scores/info for action.
            low_retrieval_score_df(pandas.DataFrame): the data with low retrieval score.
            high_retrieval_score_df(pandas.DataFrame): the data with high retrieval score.
            aml_deployment_id(str): aml deployment id for the action.

        Returns:
            LowRetreivalScoreIndexAction: the generated low retrieval score index action.
        """
        query_intention = get_query_intention(low_retrieval_score_df[PROMPT_COLUMN]) if self.llm_summary_enabled == "true" else DEFAULT_TOPIC_NAME  # noqa: E501

        positive_samples = generate_index_action_samples(high_retrieval_score_df, False, self.max_positive_sample_size)  # noqa: E501
        negative_samples = generate_index_action_samples(low_retrieval_score_df, True, self.max_positive_sample_size)

        index_content = low_retrieval_score_df.iloc[0][INDEX_CONTENT_COLUMN]
        return LowRetreivalScoreIndexAction(self.index_id,
                                            index_content,
                                            metric,
                                            confidence_score,
                                            query_intention,
                                            aml_deployment_id,
                                            os.environ.get("AZUREML_RUN_ID", None),
                                            positive_samples,
                                            negative_samples)
