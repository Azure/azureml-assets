# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for action."""

import pytest
import pandas as pd
import json
import hashlib
from pyspark.sql import Row
from datetime import datetime
from unittest.mock import patch, Mock
from action_analyzer.contracts.llm_client import LLMClient
from action_analyzer.contracts.actions.action import ActionType, Action
from action_analyzer.contracts.action_sample import ActionSample
from action_analyzer.contracts.actions.low_retrieval_score_index_action import LowRetrievalScoreIndexAction
from action_analyzer.contracts.actions.metrics_violation_index_action import MetricsViolationIndexAction
from action_analyzer.contracts.action_detectors.low_retrieval_score_index_action_detector import (
    LowRetrievalScoreIndexActionDetector
)
from action_analyzer.contracts.action_detectors.metrics_violation_index_action_detector import (
    MetricsViolationIndexActionDetector
)
from action_analyzer.contracts.utils import detector_utils
from shared_utilities.span_tree_utils import (
    SpanTree,
    SpanTreeNode,
)
from shared_utilities.constants import (
    PROMPT_COLUMN,
    MODIFIED_PROMPT_COLUMN,
    COMPLETION_COLUMN,
    ROOT_SPAN_COLUMN,
    RETRIEVAL_QUERY_TYPE_COLUMN,
    RETRIEVAL_TOP_K_COLUMN,
    PROMPT_FLOW_INPUT_COLUMN,
    INDEX_SCORE_COLUMN,
    RETRIEVAL_DOC_COLUMN,
    INDEX_ID_COLUMN,
    INDEX_CONTENT_COLUMN,
    SPAN_ID_COLUMN,
    TTEST_NAME,
    P_VALUE_THRESHOLD,
    DEFAULT_TOPIC_NAME
)

@pytest.fixture
def hashed_index_id_1():
    """Return a hashed index id 1."""
    index_content_1 = '{"self": {"asset_id": "index_asset_id_1"}}'
    return hashlib.sha256(index_content_1.encode('utf-8')).hexdigest()

@pytest.fixture
def df_with_root_span():
    """Return a dataframe for testing."""
    # The df contains 5 rows with columns [prompt, completion, root_span]
    # The root_span is structured like this:
    # 1
    # |-> 2 -> 3 (index_1 retrieval)
    # |-> 4 -> 5 (index_1 retrieval)
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

    retrieval_attributes = json.dumps({"retrieval.query": "retrieval_query_3", "retrieval.documents": json.dumps(documents)})
    s3 = SpanTreeNode(
        Row(trace_id="01", span_id="3", parent_id="2", start_time=datetime(2024, 2, 12, 9, 15, 0),
            end_time=datetime(2024, 2, 12, 9, 20, 0), span_type="Retrieval", attributes=retrieval_attributes)
    )
    s4 = SpanTreeNode(
        Row(trace_id="01", span_id="4", parent_id="1", start_time=datetime(2024, 2, 12, 9, 5, 0),
            end_time=datetime(2024, 2, 12, 9, 40, 0),
            attributes=lookup_attributes))
    
    retrieval_attributes = json.dumps({"retrieval.query": "retrieval_query_5", "retrieval.documents": json.dumps(documents)})
    s5 = SpanTreeNode(
        Row(trace_id="01", span_id="5", parent_id="4", start_time=datetime(2024, 2, 12, 9, 15, 0),
            end_time=datetime(2024, 2, 12, 9, 20, 0), span_type="Retrieval", attributes=retrieval_attributes)
    )
    spans = [s1, s2, s3, s4, s5]
    root_span = SpanTree(spans).to_json_str()

    df = pd.DataFrame(columns=[
        PROMPT_COLUMN,
        COMPLETION_COLUMN,
        ROOT_SPAN_COLUMN
    ])

    for i in range(5):
        df = df.append({
            PROMPT_COLUMN: f"prompt_{i}",
            COMPLETION_COLUMN: f"completion_{i}",
            ROOT_SPAN_COLUMN: root_span
        }, ignore_index=True)
    return df


@pytest.fixture
def preprocessed_df_score():
    """Return a df with preprocessed schema."""
    df = pd.DataFrame(columns=[
        PROMPT_COLUMN,
        MODIFIED_PROMPT_COLUMN,
        COMPLETION_COLUMN,
        SPAN_ID_COLUMN,
        INDEX_ID_COLUMN,
        INDEX_CONTENT_COLUMN,
        RETRIEVAL_DOC_COLUMN,
        ROOT_SPAN_COLUMN,
        RETRIEVAL_QUERY_TYPE_COLUMN,
        RETRIEVAL_TOP_K_COLUMN,
        PROMPT_FLOW_INPUT_COLUMN,
        "Fluency"
    ])

    for i in range(10):
        df = df.append({
            PROMPT_COLUMN: f"prompt_{i}",
            MODIFIED_PROMPT_COLUMN: f"modified_prompt_{i}",
            COMPLETION_COLUMN: f"completion_{i}",
            SPAN_ID_COLUMN: i,
            INDEX_ID_COLUMN: "index_asset_id_1",
            INDEX_CONTENT_COLUMN: "index content",
            RETRIEVAL_DOC_COLUMN: "retrieval docs",
            ROOT_SPAN_COLUMN: 'root span',
            RETRIEVAL_QUERY_TYPE_COLUMN: 'retrieval query type',
            RETRIEVAL_TOP_K_COLUMN: 3,
            PROMPT_FLOW_INPUT_COLUMN: 'flow input',
            "Fluency": 1 if i < 5 else 5  # half good half bad
        }, ignore_index=True)
    return df


@pytest.mark.unit
class TestActionDetector():
    """Test class for action detector."""

    def test_low_retrieval_score_index_action_detector_preprocess(self, df_with_root_span, hashed_index_id_1):
        """Test LowRetrievalScoreIndexActionDetector preprocess."""
        detector = LowRetrievalScoreIndexActionDetector(hashed_index_id_1, ["Fluency", "Coherence"], "true")
        preprocessed_df = detector.preprocess_data(df_with_root_span)

        column_names = list(preprocessed_df.columns.values)
        assert SPAN_ID_COLUMN in column_names
        assert INDEX_ID_COLUMN in column_names
        assert INDEX_CONTENT_COLUMN in column_names
        assert MODIFIED_PROMPT_COLUMN in column_names
        assert RETRIEVAL_DOC_COLUMN in column_names
        assert INDEX_SCORE_COLUMN in column_names
        assert RETRIEVAL_QUERY_TYPE_COLUMN in column_names
        assert RETRIEVAL_TOP_K_COLUMN in column_names
        assert PROMPT_FLOW_INPUT_COLUMN in column_names

        assert len(preprocessed_df.index) == 10
        assert preprocessed_df.iloc[0][MODIFIED_PROMPT_COLUMN] == "retrieval_query_3"
        assert preprocessed_df.iloc[0][SPAN_ID_COLUMN] == "3"
        assert preprocessed_df.iloc[1][MODIFIED_PROMPT_COLUMN] == "retrieval_query_5"
        assert preprocessed_df.iloc[1][SPAN_ID_COLUMN] == "5"

    def test_low_retrieval_score_index_action_detector_detect_with_action(self,
                                                                          preprocessed_df_score,
                                                                          hashed_index_id_1):
        """Test LowRetrievalScoreIndexAction detect with action detected."""
        detector = LowRetrievalScoreIndexActionDetector(hashed_index_id_1, ["Fluency"], "false")
        with patch("action_analyzer.contracts.utils.detector_utils._query_llm_score") as retrieval_score, \
             patch("action_analyzer.contracts.utils.detector_utils._post_process_retrieval_results") as post_score, \
             patch.object(LLMClient, "__new__"):
            retrieval_score.return_value = "fake value"
            # All retrieval scores are 1. All bad queries are due to index issue.
            post_score.return_value = 1
            fake_llm_client = LLMClient("workspace_connection_arm_id", "model_deployment_name")

            actions = detector.detect(preprocessed_df_score, fake_llm_client)
            assert len(actions) == 1
            assert actions[0].action_type == ActionType.LOW_RETRIEVAL_SCORE_INDEX_ACTION
            assert actions[0].confidence_score == 0.9
            assert actions[0].query_intention == DEFAULT_TOPIC_NAME
            assert len(actions[0].negative_samples) == 5

    def test_low_retrieval_score_index_action_detector_detect_no_action(self,
                                                                        preprocessed_df_score,
                                                                        hashed_index_id_1):
        """Test LowRetrievalScoreIndexAction detect with no action."""
        detector = LowRetrievalScoreIndexActionDetector(hashed_index_id_1, ["Fluency"], "false")
        with patch("action_analyzer.contracts.utils.detector_utils._query_llm_score") as retrieval_score, \
             patch("action_analyzer.contracts.utils.detector_utils._post_process_retrieval_results") as post_score, \
             patch.object(LLMClient, "__new__"):
            retrieval_score.return_value = "fake value"
            # All retrieval scores are 5. All bad queries are not due to index issue.
            post_score.return_value = 5
            fake_llm_client = LLMClient("workspace_connection_arm_id", "model_deployment_name")

            actions = detector.detect(preprocessed_df_score, fake_llm_client)
            assert len(actions) == 0

    def test_metrics_violation_index_action_detector_preprocess(self, df_with_root_span, hashed_index_id_1):
        """Test MetricsViolationIndexActionDetector preprocess."""
        detector = MetricsViolationIndexActionDetector(hashed_index_id_1,
                                                       ["Fluency", "Coherence"],
                                                       TTEST_NAME,
                                                       P_VALUE_THRESHOLD,
                                                       "true")
        preprocessed_df = detector.preprocess_data(df_with_root_span)

        column_names = list(preprocessed_df.columns.values)
        assert SPAN_ID_COLUMN in column_names
        assert INDEX_ID_COLUMN in column_names
        assert INDEX_CONTENT_COLUMN in column_names
        assert MODIFIED_PROMPT_COLUMN in column_names
        assert RETRIEVAL_DOC_COLUMN in column_names
        assert INDEX_SCORE_COLUMN in column_names
        assert RETRIEVAL_QUERY_TYPE_COLUMN in column_names
        assert RETRIEVAL_TOP_K_COLUMN in column_names
        assert PROMPT_FLOW_INPUT_COLUMN in column_names

        assert len(preprocessed_df.index) == 10
        assert preprocessed_df.iloc[0][MODIFIED_PROMPT_COLUMN] == "retrieval_query_3"
        assert preprocessed_df.iloc[0][SPAN_ID_COLUMN] == "3"
        assert preprocessed_df.iloc[1][MODIFIED_PROMPT_COLUMN] == "retrieval_query_5"
        assert preprocessed_df.iloc[1][SPAN_ID_COLUMN] == "5"

    def test_metrics_violation_index_action_detector_with_action(self, preprocessed_df_score, hashed_index_id_1):
        """Test MetricsViolationIndexActionDetector detect with action detected."""
        detector = MetricsViolationIndexActionDetector(hashed_index_id_1,
                                                       ["Fluency"],
                                                       TTEST_NAME,
                                                       P_VALUE_THRESHOLD,
                                                       "false")
        with patch("action_analyzer.contracts.utils.detector_utils._query_llm_score") as retrieval_score, \
             patch("action_analyzer.contracts.utils.detector_utils._post_process_retrieval_results") as post_score, \
             patch.object(LLMClient, "__new__"):
            retrieval_score.return_value = "fake value"
            # E2E metric score is 1, retrieval score is 1. E2E metrics score is 5, retrieval score is 5. T-test should show significance.  # noqa
            post_score.side_effect = [1, 1, 1, 1, 1, 5, 5, 5, 5, 5]
            fake_llm_client = LLMClient("workspace_connection_arm_id", "model_deployment_name")

            actions = detector.detect(preprocessed_df_score, fake_llm_client)
            assert len(actions) == 1
            assert actions[0].action_type == ActionType.METRICS_VIOLATION_INDEX_ACTION
            assert actions[0].confidence_score > 0.95
            assert actions[0].query_intention == DEFAULT_TOPIC_NAME
            assert len(actions[0].negative_samples) == 5
            assert len(actions[0].positive_samples) == 5


    def test_metrics_violation_index_action_detector_no_action(self, preprocessed_df_score, hashed_index_id_1):
        """Test MetricsViolationIndexActionDetector detect with no action."""
        detector = MetricsViolationIndexActionDetector(hashed_index_id_1,
                                                       ["Fluency"],
                                                       TTEST_NAME,
                                                       P_VALUE_THRESHOLD,
                                                       "false")
        with patch("action_analyzer.contracts.utils.detector_utils._query_llm_score") as retrieval_score, \
             patch("action_analyzer.contracts.utils.detector_utils._post_process_retrieval_results") as post_score, \
             patch.object(LLMClient, "__new__"):
            retrieval_score.return_value = "fake value"
            # All retrieval scores are 1. T-test should not show significance.
            post_score.return_value = 1
            fake_llm_client = LLMClient("workspace_connection_arm_id", "model_deployment_name")

            actions = detector.detect(preprocessed_df_score, fake_llm_client)
            assert len(actions) == 0
