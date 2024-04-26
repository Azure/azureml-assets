# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action sample class."""

from shared_utilities.io_utils import np_encoder
from action_analyzer.contracts.utils.action_utils import convert_to_camel_case


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


class IndexActionSample(ActionSample):
    """Index Action sample class."""

    def __init__(self,
                 question: str,
                 index_query: str,
                 answer: str,
                 lookup_score: float,
                 debugging_info: str,
                 retrieval_query_type: str,
                 retrieval_top_k: int,
                 prompt_flow_input: str) -> None:
        """Create an index action sample.

        Args:
            question(str): the input question of the flow.
            index_query(str): the index query for document retrieval.
            answer(str): the output answer of the flow
            lookup_score(float): the retrieval document look up score.
            debugging_info(str): the json string of debugging info in a span tree structure.
            retrieval_query_type(str): the retrieval query type in the retrieval span.
            retrieval_top_k(int): the retrieval top k value in the retrieval span.
            prompt_flow_input(str): the json str of prompt flow input.
        """
        self.lookup_score = lookup_score
        self.index_query = index_query
        self.retrieval_query_type = retrieval_query_type
        self.retrieval_top_k = retrieval_top_k
        super().__init__(question,
                         answer,
                         debugging_info,
                         prompt_flow_input)
