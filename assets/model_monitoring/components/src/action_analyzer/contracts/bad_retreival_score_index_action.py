# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""BadRetreivalScoreIndexAction Class."""

from action_analyzer.contracts.action import ActionType, Action
from shared_utilities.constants import (
    ACTION_DESCRIPTION
)

class BadRetreivalScoreIndexAction(Action):
    """Bad retrieval score index action class."""

    def __init__(self,
                 index_id: str,
                 index_content: str,
                 confidence_score: str,
                 query_intention: str,
                 deployment_id: str,
                 run_id: str,
                 positive_samples: list[str],
                 negative_samples: list[str],
                 index_name=None):
        """Create a bad retrieval score index action.

        Args:
            index_id(str): the index asset id.
            index_content(str): the index content.
            confidence_score(float): the confidence score of the action.
            query_intention(str): the query intention of the action.
            deployment_id(str): the azureml deployment id of the action.
            run_id(str): the azureml run id which generates the action.
            positive_samples(list[str]): list of positive samples of the action.
            negative_samples(list[str]): list of negative samples of the action.
            index_name(str): (optional) index name if index asset id does not exist.
        """
        self.IndexAssetId = index_id
        self.IndexName = index_name
        self.IndexContent = index_content
        description = ACTION_DESCRIPTION.replace("{index_id}", index_id)
        super().__init__(ActionType.BAD_RETRIEVAL_SCORE_INDEX_ACTION,
                         description,
                         confidence_score,
                         query_intention,
                         deployment_id,
                         run_id,
                         positive_samples,
                         negative_samples)