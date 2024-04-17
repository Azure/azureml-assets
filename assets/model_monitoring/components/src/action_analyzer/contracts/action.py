# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action Class."""

import datetime
import uuid
import json
from enum import Enum
from abc import ABC, abstractmethod
from action_analyzer.utils.utils import convert_to_camel_case

class ActionSample:
    """Action sample class."""

    def __init__(self,
                 question: str,
                 answer: str,
                 lookup_score: float,
                 debugging_info: str,
                 retrieval_query_type: str,
                 retrieval_top_k: int,
                 prompt_flow_input: str) -> None:
        """Create an action sample.

        Args:
            question(str): the input question of the flow.
            answer(str): the output answer of the flow
            lookup_score(float): the retrieval document look up score.
            debugging_info(str): the json string of debugging info in a span tree structure.
            retrieval_query_type(str): the retrieval query type in the retrieval span.
            retrieval_top_k(int): the retrieval top k value in the retrieval span.
            prompt_flow_input(str): the json str of prompt flow input.
        """
        self.question = question
        self.answer = answer
        self.lookup_score = lookup_score
        self.debugging_info = debugging_info
        self.retrieval_query_type = retrieval_query_type
        self.retrieval_top_k = retrieval_top_k
        self.prompt_flow_input = prompt_flow_input


    def to_json_str(self) -> str:
        """Convert an action sample object to json string."""
        attribute_dict =  self.__dict__
        json_out = {}
        for key, val in attribute_dict.items():
            json_out[convert_to_camel_case(key)] = val

        return json.dumps(json_out)


class ActionType(Enum):
    """Action type."""

    METRICS_VIOLATION_INDEX_ACTION = 1
    BAD_RETRIEVAL_SCORE_INDEX_ACTION = 2


class Action():
    """Action class."""

    def __init__(self,
                 action_type: ActionType, 
                 description: str,
                 confidence_score: float,
                 query_intention: str,
                 deployment_id: str,
                 run_id: str,
                 positive_samples: list[str],
                 negative_samples: list[str]) -> None:
        """Create an action.

        Args:
            action_type(ActionType): the action type.
            description(str): the description of the action.
            confidence_score(float): the confidence score of the action.
            query_intention(str): the query intention of the action.
            deployment_id(str): the azureml deployment id of the action.
            run_id(int): the azureml run id which generates the action.
            positive_samples(list[str]): list of positive samples of the action.
            negative_samples(list[str]): list of negative samples of the action.
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


    def to_json_str(self) -> str:
        """Convert an action object to json str."""
        attribute_dict =  self.__dict__
        json_out = {}
        for key, val in attribute_dict.items():
            if key == "action_type":
                json_out[convert_to_camel_case(key)] = val.name
            # serialize the samples
            elif key.endswith("_samples"):
                json_val = str([v.to_json_str() for v in val])
                json_out[convert_to_camel_case(key)] = json_val
            else:
                json_out[convert_to_camel_case(key)] = val
        return json.dumps(json_out)


class DebuggingInfo:
    """Debugging info class."""

    def __init__(self,
                 span_id: str,
                 index_content: str,
                 index_id: str,
                 query: str,
                 retrieval_documents: str,
                 retrieval_score: float,
                 retrieval_query_type: str,
                 retrieval_top_k: int,
                 prompt_flow_input: str) -> None:
        """Create debugging info.        
        
        Args:
            span_id(str): the span id of te retreival span.
            index_content(str): the azureml index content.
            index_id(str): the index asset id.
            query(str): the input query (can be modified) in the retrieval span.
            retrieval_documents(str): the retreival documents (joint by splitter).
            retrieval_score(float): the built-in retrieval score from the model.
            retrieval_query_type(str): the retrieval query type in the retrieval span. e.g. Hybrid (vector + keyword)
            retrieval_top_k(int): the retrieval top k value in the retrieval span.
            prompt_flow_input(str): the json str of prompt flow input.
        """
        self.span_id = span_id
        self.index_content = index_content
        self.index_id = index_id
        self.query = query
        self.retrieval_documents = retrieval_documents
        self.retrieval_score = retrieval_score
        self.retrieval_query_type = retrieval_query_type
        self.retrieval_top_k = retrieval_top_k
        self.prompt_flow_input = prompt_flow_input


    def to_json_str(self) -> str:
        """Convert a debugging info object to json str."""
        return json.dumps(self.__dict__)
