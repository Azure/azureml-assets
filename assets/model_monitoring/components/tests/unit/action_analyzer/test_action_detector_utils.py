# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for action detector utils."""

import pytest
import pandas as pd
import json
from pyspark.sql import Row
from datetime import datetime
from action_analyzer.contracts.action_sample import IndexActionSample
from action_analyzer.contracts.utils.detector_utils import *
from action_analyzer.action_detector_component.run import *
from shared_utilities.span_tree_utils import (
    SpanTree,
    SpanTreeNode,
)
from shared_utilities.constants import (
    PROMPT_COLUMN,
    MODIFIED_PROMPT_COLUMN,
    COMPLETION_COLUMN,
    INDEX_SCORE_LLM_COLUMN,
    ROOT_SPAN_COLUMN,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    PROMPT_FLOW_INPUT_COLUMN
)

@pytest.fixture
def df_for_action_sample():
    """Return a df for generating action sample."""
    df = pd.DataFrame(columns=[
        PROMPT_COLUMN,
        MODIFIED_PROMPT_COLUMN,
        COMPLETION_COLUMN,
        INDEX_SCORE_LLM_COLUMN,
        ROOT_SPAN_COLUMN,
        RETRIEVAL_QUERY_TYPE_COLUMN,
        RETRIEVAL_TOP_K_COLUMN,
        PROMPT_FLOW_INPUT_COLUMN
    ])

    for i in range(10):
        df = df.append({
            PROMPT_COLUMN: 'Prompt',
            MODIFIED_PROMPT_COLUMN: 'Modified Prompt',
            COMPLETION_COLUMN: 'Completion',
            INDEX_SCORE_LLM_COLUMN: i,
            ROOT_SPAN_COLUMN: 'Root Span',
            RETRIEVAL_QUERY_TYPE_COLUMN: 'Query Type',
            RETRIEVAL_TOP_K_COLUMN: 5,
            PROMPT_FLOW_INPUT_COLUMN: 'Flow Input'
        }, ignore_index=True)
    return df

@pytest.fixture
def retrieval_root_span():
    """Return a root span for retrieval."""
    # The tree is structured like this:
    # 1
    # |-> 2 -> 3 (retrieval)
    # |-> 4 -> 5 (retreival)
    mlindex_content = '{"self": {"asset_id": "index_asset_id_1"}}'
    index_input = json.dumps({"mlindex_content": mlindex_content})
    attribute_str = json.dumps({"inputs": index_input})

    s1 = SpanTreeNode(
        Row(trace_id="01", span_id="1", parent_id=None, start_time=datetime(2024, 2, 12, 9, 0, 0),
            end_time=datetime(2024, 2, 12, 10, 5, 0)))
    s2 = SpanTreeNode(
        Row(trace_id="01", span_id="2", parent_id="1", start_time=datetime(2024, 2, 12, 9, 5, 0),
            end_time=datetime(2024, 2, 12, 9, 40, 0),
            attributes=attribute_str))
    s3 = SpanTreeNode(
        Row(trace_id="01", span_id="3", parent_id="2", start_time=datetime(2024, 2, 12, 9, 15, 0),
            end_time=datetime(2024, 2, 12, 9, 20, 0), span_type="Retrieval")
    )

    mlindex_content = '{"self": {"asset_id": "index_asset_id_2"}}'
    index_input = json.dumps({"mlindex_content": mlindex_content})
    attribute_str = json.dumps({"inputs": index_input})
    s4 = SpanTreeNode(
        Row(trace_id="01", span_id="4", parent_id="1", start_time=datetime(2024, 2, 12, 9, 25, 0),
            end_time=datetime(2024, 2, 12, 9, 30, 0),
            attributes=attribute_str))
    s5 = SpanTreeNode(
        Row(trace_id="01", span_id="5", parent_id="4", start_time=datetime(2024, 2, 12, 9, 45, 0),
            end_time=datetime(2024, 2, 12, 9, 50, 0), span_type="Retrieval")
    )

    spans = [s1, s2, s3, s4, s5]
    return SpanTree(spans).to_json_str()


@pytest.fixture
def root_span():
    """Return a root span with all debugging info."""
    # The tree is structured like this:
    # 1
    # |-> 2 -> 3 (retrieval)


    s1 = SpanTreeNode(
        Row(trace_id="01", span_id="1", parent_id=None, start_time=datetime(2024, 2, 12, 9, 0, 0),
            end_time=datetime(2024, 2, 12, 10, 5, 0),
            attributes=json.dumps({"inputs": "prompt_flow_input"})))

    mlindex_content = '{"self": {"asset_id": "index_asset_id_1"}}'
    index_input = json.dumps({"mlindex_content": mlindex_content,
                              "query_type": "Hybrid (vector + keyword)",
                              "top_k": 3})
    lookup_attributes = json.dumps({"inputs": index_input})
    s2 = SpanTreeNode(
        Row(trace_id="01", span_id="2", parent_id="1", start_time=datetime(2024, 2, 12, 9, 5, 0),
            end_time=datetime(2024, 2, 12, 9, 40, 0),
            attributes=lookup_attributes))
    documents = []
    for i in range(3):
        documents.append({"document.content": f"doc_{i}", "document.score": "0.5"})

    retrieval_attributes = json.dumps({"retrieval.query": "query","retrieval.documents": json.dumps(documents)})
    s3 = SpanTreeNode(
        Row(trace_id="01", span_id="3", parent_id="2", start_time=datetime(2024, 2, 12, 9, 15, 0),
            end_time=datetime(2024, 2, 12, 9, 20, 0), span_type="Retrieval", attributes=retrieval_attributes)
    )
    spans = [s1, s2, s3]
    return SpanTree(spans).to_json_str()

@pytest.mark.unit
class TestDetectorUtils:
    """Test class for action."""

    @pytest.mark.parametrize(
        "index_content, expected_index_id", [
            ('{"self": {"asset_id": "index_asset_id"}}', "index_asset_id"),
            ('{"index": {"index": "index_name"}}', "index_name"),
            ('{}', None),
            ('invalid_yaml', None)
        ])
    def test_get_index_id_from_index_content(self, index_content, expected_index_id):
        """Test get_index_id_from_index_content function."""
        result = get_index_id_from_index_content(index_content)
        assert result == expected_index_id

    def test_get_missed_metrics(self):
        """Test get_missed_metrics function."""
        violated_metrics = ["Fluency", "Coherence", "Relevance", "Groundedness", "Similarity", "other metrics"]
        column_names = ["query", "completion", "Fluency", "Coherence"]
        missed_metrics = get_missed_metrics(violated_metrics, column_names)
        assert missed_metrics == ["Relevance", "Groundedness", "Similarity"]

    def test_get_violated_metrics_success(self):
        """Test get violated metrics from gsq output."""
        # "AggregatedFluencyPassRate": {
        #     "value": "0.92",
        #     "threshold": "0.9",
        # }
        # "AggregatedCoherencePassRate": {
        #     "value": "0.86",
        #     "threshold": "0.9",
        # }
        # "AggregatedRelevancePassRate": {
        #     "value": "0.77",
        #     "threshold": "0.9",
        # }
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../")
        input_url = f"{tests_path}/unit/action_analyzer/resources/"
        print(input_url)
        violated_metrics = get_violated_metrics(input_url, "gsq-signal")
        assert violated_metrics == ["Coherence", "Relevance"]

    def test_get_violated_metrics_fail(self):
        """Test get violated metrics from empty gsq output."""
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../")
        input_url = f"{tests_path}/unit/action_analyzer/resources/"
        violated_metrics = get_violated_metrics(input_url, "empty")
        assert violated_metrics == []

    def test_generate_index_action_samples_negative(self, df_for_action_sample):
        """Test generate_index_action_samples function for negative samples."""
        action_samples = generate_index_action_samples(df_for_action_sample, True)
        assert len(action_samples) == 10
        assert type(action_samples[0]) == IndexActionSample
        # the score is not sorted for negative samples
        assert action_samples[0].lookup_score == 0
        assert action_samples[9].lookup_score == 9

    def test_generate_index_action_samples_positive(self, df_for_action_sample):
        """Test generate_index_action_samples function for positive samples."""
        action_samples = generate_index_action_samples(df_for_action_sample, False)
        assert len(action_samples) == 10
        assert type(action_samples[0]) == IndexActionSample
        # the score is sorted in descent order for positive samples
        assert action_samples[0].lookup_score == 9
        assert action_samples[9].lookup_score == 0

    def test_parse_index_id(self, retrieval_root_span):
        """Test parse_index_id function from the root span."""
        # The tree is structured like this:
        # 1
        # |-> 2 -> 3 (retrieval)
        # |-> 4 -> 5 (retreival)
        index_ids = parse_index_id(retrieval_root_span)
        assert len(index_ids) == 2
        assert "index_asset_id_1" in index_ids
        assert "index_asset_id_2" in index_ids


    def test_get_unique_indexes(self, retrieval_root_span):
        """Test get_unique_indexes function from the root span."""
        # The tree is structured like this:
        # 1
        # |-> 2 -> 3 (retrieval: index_asset_1)
        # |-> 4 -> 5 (retreival: index_asset_2)
        df = pd.DataFrame(columns=[ROOT_SPAN_COLUMN])
        for i in range(10):
            df = df.append({
                ROOT_SPAN_COLUMN: retrieval_root_span
            }, ignore_index=True)

        unique_indexes = get_unique_indexes(df)
        assert len(unique_indexes) == 2
        assert "index_asset_id_1" in unique_indexes
        assert "index_asset_id_2" in unique_indexes


    def test_parse_debugging_info(self, root_span):
        """Test parse_debugging_info function from the root span."""
        extra_feilds = json.loads(parse_debugging_info(root_span, "index_asset_id_1")[0])
        print(extra_feilds)
        assert extra_feilds[SPAN_ID_COLUMN] == "3"
        assert extra_feilds[INDEX_CONTENT_COLUMN] == '{"self": {"asset_id": "index_asset_id_1"}}'
        assert extra_feilds[INDEX_ID_COLUMN] == "index_asset_id_1"
        assert extra_feilds[MODIFIED_PROMPT_COLUMN] == "query"
        assert extra_feilds[RETRIEVAL_DOC_COLUMN] == "doc_0#<Splitter>#doc_1#<Splitter>#doc_2"
        assert extra_feilds[INDEX_SCORE_COLUMN] == 0.5
        assert extra_feilds[RETRIEVAL_QUERY_TYPE_COLUMN] == "Hybrid (vector + keyword)"
        assert extra_feilds[RETRIEVAL_TOP_K_COLUMN] == 3
        assert extra_feilds[PROMPT_FLOW_INPUT_COLUMN] == "prompt_flow_input"
