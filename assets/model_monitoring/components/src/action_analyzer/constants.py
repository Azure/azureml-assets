# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# action analyzer constants
PROMPT_COLUMN = "prompt"
COMPLETION_COLUMN = "completion"
CONTEXT_COLUMN = "context"
TRACE_ID_COLUMN = "trace_id"
ROOT_PROMPT_COLUMN = "root_prompt"
VIOLATED_METRICS_COLUMN = "violated_metrics"
INDEX_CONTENT_COLUMN = "index_content"
INDEX_SCORE_COLUMN = "index_score"
INDEX_ID_COLUMN = "index_id"
ROOT_SPAN_COLUMN = "root_span"
GOOD_GROUP_NAME = "{metrics}_good"
BAD_GROUP_NAME = "{metrics}_bad"
CONFIDENCE_SCORE_COLUMN = "confidence_score"
ACTION_ID_COLUMN = "action_id"
RETRIEVAL_QUERY_TYPE_COLUMN = "retrieval_query_type"
RETRIEVAL_TOP_K_COLUMN = "retrieval_top_k"
DEFAULT_TOPIC_NAME = "disparate"
GROUP_COLUMN = "query_group"
QUERY_INTENTION_COLUMN = "query_intention"
ACTION_METRICS_COLUMN = "action_metrics"
PROPERTIES_COLUMN = "action_analyzer_properties"
TTEST_GROUP_ID_COLUMN = "ttest_group_id"
TRACE_ID_LIST_COLUMN = "trace_id_list"
ACTION_QUERY_INTENTION_COLUMN = "action_query_intention"
ACTION_ANALYZER_ACTION_TAG = "momo_action_analyzer_has_action"

GSQ_METRICS_LIST = ["Fluency", "Coherence", "Groundedness", "Relevance", "Similarity"]
GOOD_METRICS_VALUE = 5
METRICS_VIOLATION_THRESHOLD = 4
RETRIEVAL_SPAN_TYPE = "Retrieval"
TEXT_SPLITTER = "#<Splitter>#"

GROUP_TOPIC_MIN_SAMPLE_SIZE = 10
P_VALUE_THRESHOLD = 0.05
MEAN_THRESHOLD = 3

INDEX_ACTION_TYPE = "Index Action"
ACTION_DESCRIPTION = "The application's response quality is low due to suboptimal index retrieval. Please update the index with ID '{index_id}' to address this issue."  # noqa
MAX_SAMPLE_SIZE = 20
INVALID_METRICS_SCORE = -1
