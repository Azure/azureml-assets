# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""LowRetrievalScoreIndexAction Class."""

from typing import List
from action_analyzer.contracts.actions.action import ActionType, Action
from action_analyzer.contracts.action_sample import IndexActionSample
from shared_utilities.constants import (
    ACTION_DESCRIPTION
)


class LowRetrievalScoreIndexAction(Action):
    """Low retrieval score index action class."""

    def __init__(self,
                 index_asset_id: str,
                 index_content: str,
                 violated_metrics: str,
                 confidence_score: str,
                 query_intention: str,
                 deployment_id: str,
                 run_id: str,
                 positive_samples: List[IndexActionSample],
                 negative_samples: List[IndexActionSample],
                 index_name=None) -> None:
        """Create a low retrieval score index action.

        Args:
            index_asset_id(str): the index asset id.
            index_content(str): the index content.
            violated_metrics(str): violated metrics in comma-separated string format.
            confidence_score(float): the confidence score of the action.
            query_intention(str): the query intention of the action.
            deployment_id(str): the azureml deployment id of the action.
            run_id(str): the azureml run id which generates the action.
            positive_samples(List[IndexActionSample]): list of positive samples of the action.
            negative_samples(List[IndexActionSample]): list of negative samples of the action.
            index_name(str): (optional) index name if index asset id does not exist.
        """
        self.index_asset_id = index_asset_id
        self.index_name = index_name
        self.index_content = index_content
        self.violated_metrics = violated_metrics
        description = ACTION_DESCRIPTION.replace("{index_id}", index_asset_id)
        super().__init__(ActionType.LOW_RETRIEVAL_SCORE_INDEX_ACTION,
                         description,
                         confidence_score,
                         query_intention,
                         deployment_id,
                         run_id,
                         positive_samples,
                         negative_samples)

    def to_summary_json(self, action_output_folder) -> dict:
        """Get the meta data for action summary.

        Args:
            action_output_folder(str): output folder path for actions.

        Returns:
            dict: action summary with metadata.
        """
        summary_json = super().to_summary_json(action_output_folder)
        summary_json["ViolatedMetrics"] = self.violated_metrics
        return summary_json
