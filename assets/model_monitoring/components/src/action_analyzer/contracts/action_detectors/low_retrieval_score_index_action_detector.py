# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Low retrieval score index action detector class."""

import os
import pandas
from typing import List
from action_analyzer.contracts.action_detectors.action_detector import ActionDetector
from action_analyzer.contracts.actions.action import Action
from action_analyzer.contracts.actions.low_retrieval_score_index_action import LowRetrievalScoreIndexAction
from action_analyzer.contracts.llm_client import LLMClient
from action_analyzer.contracts.utils.detector_utils import (
    extract_retrieval_info_from_root_span,
    get_retrieval_score,
    get_query_intention,
    generate_index_action_samples
)
from shared_utilities.constants import (
    INDEX_ID_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    METRICS_VIOLATION_THRESHOLD,
    GOOD_METRICS_THRESHOLD,
    DEFAULT_TOPIC_NAME,
    INDEX_CONTENT_COLUMN,
    PROMPT_COLUMN,
    LOW_RETRIEVAL_SCORE_THRESHOLD,
    HIGH_RETRIEVAL_SCORE_THRESHOLD,
    INVALID_LLM_SCORE
)


LOW_RETRIEVAL_SCORE_QUERY_RATIO_THRESHOLD = 0.5
LOW_RETRIEVAL_SCORE_INDEX_ACTION_CONFIDENCE = 0.9


class LowRetrievalScoreIndexActionDetector(ActionDetector):
    """Low retrieval score index action detector class."""

    def __init__(self,
                 index_id: str,
                 violated_metrics: List[str],
                 query_intention_enabled: str,
                 preprocessed_data: pandas.DataFrame=pandas.DataFrame()) -> None:
        """Create a low retrieval score index action detector.

        Args:
            index_id(str): the hashed index id.
            violated_metrics(List[str]): violated e2e metrics
            query_intention_enabled(str): enable llm generated summary. Accepted values: true or false.
            preprocessed_data(pandas.DataFrame): (Optional) preprocessed data. If passed, skip the preprocess step.
        """
        self.index_id = index_id
        self.violated_metrics = violated_metrics
        super().__init__(query_intention_enabled, preprocessed_data)

    def preprocess_data(self, df: pandas.DataFrame):
        """Preprocess the data for action detector. Convert the dataframe from trace level to span level.

        Args:
            df(pandas.DataFrame): input pandas dataframe.
        """
        try:
            if not self.preprocessed_data.empty:
                print("Preprocessed data is available. Skip executing preprocess.")
                return
            
            print("Start to run LowRetrievalScoreIndexActionDetector.")
            preprocessed_df = extract_retrieval_info_from_root_span(df, self.index_id)

            # get llm retrieval score
            preprocessed_df[INDEX_SCORE_LLM_COLUMN] = preprocessed_df.apply(get_retrieval_score,
                                                                            axis=1,
                                                                            args=(llm_client,))
            self.preprocessed_data = preprocessed_df[preprocessed_df[INDEX_SCORE_LLM_COLUMN] != INVALID_LLM_SCORE]

        except Exception as e:
            print("LowRetrievalScoreIndexActionDetector preprocess failed with error", e)

    def detect(self, llm_client: LLMClient, aml_deployment_id=None) -> List[Action]:
        """Detect the action.

        Args:
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.
            aml_deployment_id(str): (Optional) aml deployment id for the action.

        Returns:
            List[Action]: list of actions.
        """
        action_list = []
        if not self.df_preprocessed.empty:
            df = self.preprocessed_data
            try:
                for metric in self.violated_metrics:
                    low_retrieval_score_df = df[(df[metric] < METRICS_VIOLATION_THRESHOLD) &
                                                (df[INDEX_SCORE_LLM_COLUMN] < LOW_RETRIEVAL_SCORE_THRESHOLD)]
                    low_metric_score_df = df[df[metric] < METRICS_VIOLATION_THRESHOLD]
                    high_retrieval_score_df = df[(df[metric] >= GOOD_METRICS_THRESHOLD) &
                                                (df[INDEX_SCORE_LLM_COLUMN] >= HIGH_RETRIEVAL_SCORE_THRESHOLD)]
                    # generate action only low retrieval score query ratio above the threshold
                    low_retrieval_score_query_ratio = len(low_retrieval_score_df)/len(low_metric_score_df)
                    print(f"The low retrieval score query ratio for metric {metric}: {low_retrieval_score_query_ratio}.")
                    if low_retrieval_score_query_ratio >= LOW_RETRIEVAL_SCORE_QUERY_RATIO_THRESHOLD:
                        print(f"Generating action for metric {metric}.")
                        # use the low retrieval score query ratio as confidence
                        action = self.generate_action(llm_client,
                                                    metric,
                                                    LOW_RETRIEVAL_SCORE_INDEX_ACTION_CONFIDENCE,
                                                    low_retrieval_score_df,
                                                    high_retrieval_score_df,
                                                    aml_deployment_id)
                        print(f"Positive sample size: {len(action.positive_samples)}.")
                        print(f"Negative sample size: {len(action.negative_samples)}.")
                        action_list.append(action)
            except Exception as e:
                print("LowRetrievalScoreIndexActionDetector detect failed with error", e)
        return action_list

    def generate_action(self,
                        llm_client: LLMClient,
                        metric: str,
                        confidence_score: float,
                        low_retrieval_score_df: pandas.DataFrame,
                        high_retrieval_score_df: pandas.DataFrame,
                        aml_deployment_id: str) -> LowRetrievalScoreIndexAction:
        """Generate action from the dataframe.

        Args:
            llm_client(LLMClient): LLM client used to get some llm scores/info for action.
            metric(str): the violated metric for action.
            confidence_score(float): LLM client used to get some llm scores/info for action.
            low_retrieval_score_df(pandas.DataFrame): the data with low retrieval score.
            high_retrieval_score_df(pandas.DataFrame): the data with high retrieval score.
            aml_deployment_id(str): aml deployment id for the action.

        Returns:
            LowRetrievalScoreIndexAction: the generated low retrieval score index action.
        """
        query_intention = get_query_intention(low_retrieval_score_df[PROMPT_COLUMN].to_list(), llm_client) if self.query_intention_enabled == "true" else DEFAULT_TOPIC_NAME  # noqa: E501
        print("Got query intention: ", query_intention)
        positive_samples = generate_index_action_samples(high_retrieval_score_df, False)
        negative_samples = generate_index_action_samples(low_retrieval_score_df, True)

        index_content = low_retrieval_score_df.iloc[0][INDEX_CONTENT_COLUMN]
        index_asset_id = low_retrieval_score_df.iloc[0][INDEX_ID_COLUMN]
        print("index_asset_id: ", index_asset_id)
        return LowRetrievalScoreIndexAction(index_asset_id,
                                            index_content,
                                            metric,
                                            confidence_score,
                                            query_intention,
                                            aml_deployment_id,
                                            os.environ.get("AZUREML_RUN_ID", None),
                                            positive_samples,
                                            negative_samples)
