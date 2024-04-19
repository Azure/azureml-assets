# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Action sample class."""

import json
from action_analyzer.contracts.action import ActionSample


class IndexActionSample(ActionSample):
    """Index Action sample class."""

    def __init__(self,
                 question: str,
                 answer: str,
                 lookup_score: float,
                 debugging_info: str,
                 retrieval_query_type: str,
                 retrieval_top_k: int,
                 prompt_flow_input: str) -> None:
        """Create an index action sample.

        Args:
            question(str): the input question of the flow.
            answer(str): the output answer of the flow
            lookup_score(float): the retrieval document look up score.
            debugging_info(str): the json string of debugging info in a span tree structure.
            retrieval_query_type(str): the retrieval query type in the retrieval span.
            retrieval_top_k(int): the retrieval top k value in the retrieval span.
            prompt_flow_input(str): the json str of prompt flow input.
        """
        self.lookup_score = lookup_score
        self.retrieval_query_type = retrieval_query_type
        self.retrieval_top_k = retrieval_top_k
        super().__init__(question,
                         answer,
                         debugging_info,
                         prompt_flow_input)


class RetrievalSpanData:
    """Properties of retrieval span."""

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
        """Create retreival span data.

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
        """Convert a retrieval span data to json str."""
        return json.dumps(self.__dict__)
