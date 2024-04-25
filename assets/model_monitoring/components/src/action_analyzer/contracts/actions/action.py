# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Class."""

import os
import datetime
import uuid
import json
from enum import Enum
from action_analyzer.utils.utils import convert_to_camel_case
from shared_utilities.io_utils import np_encoder


class ActionType(Enum):
    """Action type."""

    METRICS_VIOLATION_INDEX_ACTION = 1
    LOW_RETRIEVAL_SCORE_INDEX_ACTION = 2


class ActionSample:
    """Action sample class."""

    def __init__(self,
                 question: str,
                 answer: str,
                 debugging_info: str,
                 prompt_flow_input: str) -> None:
        """Create an action sample.

        Args:
            question(str): the input question of the flow.
            answer(str): the output answer of the flow.
            debugging_info(str): the json string of debugging info in a span tree structure.
            prompt_flow_input(str): the json str of prompt flow input.
        """
        self.question = question
        self.answer = answer
        self.debugging_info = debugging_info
        self.prompt_flow_input = prompt_flow_input

    def to_json_str(self) -> str:
        """Convert an action sample object to json string."""
        attribute_dict = self.__dict__
        json_out = {}
        for key, val in attribute_dict.items():
            json_out[convert_to_camel_case(key)] = val
        return json.dumps(json_out, default=np_encoder)


class Action():
    """Action class."""

    def __init__(self,
                 action_type: ActionType,
                 description: str,
                 confidence_score: float,
                 query_intention: str,
                 deployment_id: str,
                 run_id: str,
                 positive_samples: list[ActionSample],
                 negative_samples: list[ActionSample]) -> None:
        """Create an action.

        Args:
            action_type(ActionType): the action type.
            description(str): the description of the action.
            confidence_score(float): the confidence score of the action.
            query_intention(str): the query intention of the action.
            deployment_id(str): the azureml deployment id of the action.
            run_id(int): the azureml run id which generates the action.
            positive_samples(list[ActionSample]): list of positive samples of the action.
            negative_samples(list[ActionSample]): list of negative samples of the action.
        """
        self.action_id = str(uuid.uuid4())
        self.action_type = action_type
        self.description = description
        self.confidence_score = confidence_score
        self.query_intention = query_intention
        self.creation_time = str(datetime.datetime.now())
        self.deployment_id = deployment_id
        self.run_id = run_id
        self.positive_samples = positive_samples
        self.negative_samples = negative_samples

    def to_json(self) -> dict:
        """Convert an action object to json dict."""
        attribute_dict = self.__dict__
        json_out = {}
        for key, val in attribute_dict.items():
            if key == "action_type":
                json_out["Type"] = val.name
            # serialize the samples
            elif key.endswith("_samples"):
                json_val = [v.to_json_str() for v in val]
                json_out[convert_to_camel_case(key)] = json_val
            else:
                json_out[convert_to_camel_case(key)] = val
        return json_out

    def to_summary_json(self, action_output_folder: str) -> dict:
        """Get the meta data for action summary.

        Args:
            action_output_folder(str): output folder path for actions.

        Returns:
            dict: action summary with metadata.
        """
        summary_json = {
            "ActionId": self.action_id,
            "Type": self.action_type.name,
            "Description": self.description,
            "ConfidenceScore": self.confidence_score,
            "QueryIntention": self.query_intention,
            "CreationTime": self.creation_time,
            "FilePath": os.path.join(action_output_folder, f"actions/{self.action_id}.json")
        }
        return summary_json
