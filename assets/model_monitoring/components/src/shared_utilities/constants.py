# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains constants for data drift command components."""

# Column Names of Computed Metrics Parquet
FEATURE_NAME_COLUMN = 'feature_name'
GROUP_COLUMN = 'group'
GROUP_PIVOT_COLUMN = 'group_pivot'
FEATURE_CATEGORY_COLUMN = 'data_type'
METRIC_NAME_COLUMN = 'metric_name'
METRIC_VALUE_COLUMN = 'metric_value'

# Column Names of samples index parquet.
SAMPLES_COLUMN = 'samples'
SAMPLES_NAME_COLUMN = 'samples_name'
ASSET_COLUMN = 'asset'

# Column Names of Histogram
BUCKET_COUNT_COLUMN = 'bucket_count'
CATEGORY_BUCKET_COLUMN = 'category_bucket'
FEATURE_BUCKET_COLUMN = 'feature_bucket'
FEATURE_TYPE_COLUMN = 'feature_type'
LOWER_BOUND_COLUMN = 'lower_bound'
UPPER_BOUND_COLUMN = 'upper_bound'

# Error Messages
MESSAGE_TO_CONTACT_AML = 'Please contact Microsoft support for assistance.'

# Filenames
META_FILENAME = '.meta'
METRICS_FILENAME = 'metrics.json'

# Others
CATEGORICAL_FEATURE_CATEGORY = 'categorical'
NUMERICAL_FEATURE_CATEGORY = 'numerical'
UTF8_ENCODING = 'utf-8'

# Parameters for Outputs of Output Metrics Component
BASELINE_COUNT_PARAM = 'baselineCount'
CATEGORICAL_PARAM = 'categorical'
CATEGORY_PARAM = 'category'
FEATURE_CATEGORY_PARAM = 'featureCategory'
FEATURE_FILES_PARAM = 'featureFiles'
FEATURE_NAME_PARAM = 'featureName'
HISTOGRAM_PARAM = 'histogram'
LOWER_BOUND_PARAM = 'lowerBound'
METRIC_VALUE_PARAM = 'metricValue'
METRICS_FILE_PARAM = 'metricsFile'
METRICS_PARAM = 'metrics'
METRICS_TYPE_PARAM = 'metricsType'
NUMERICAL_PARAM = 'numerical'
TARGET_COUNT_PARAM = 'targetCount'
UPPER_BOUND_PARAM = 'upperBound'
VERSION_PARAM = 'version'

# Keywords in metrics schema
AGGREGATE = 'aggregate'
SIGNAL_METRICS_GROUP = 'group'
SIGNAL_METRICS_GROUP_DIMENSION = 'group_dimension'
SIGNAL_METRICS_METRIC_NAME = 'metric_name'
SIGNAL_METRICS_METRIC_VALUE = 'metric_value'
SIGNAL_METRICS_THRESHOLD_VALUE = 'threshold_value'

# Keywords in metric output JSON schema
GROUPS = 'groups'
METRICS = "metrics"
SAMPLES_URI = "uri"
THRESHOLD = 'threshold'
TIMESERIES = "timeseries"
TIMESERIES_RUN_ID = "runId"
TIMESERIES_METRIC_NAMES = "metricNames"
TIMESERIES_METRIC_NAMES_VALUE = "value"
TIMESERIES_METRIC_NAMES_THRESHOLD = "threshold"
VALUE = 'value'
# Values for Outputs of Output Metrics Component
METADATA_VERSION = '1.0.0'

# Model types
CLASSIFICATION = 'classification'
REGRESSION = 'regression'

# Column names/values for feature attribution drift and feature importance
FEATURE_COLUMN = 'feature'
ROW_COUNT_COLUMN_NAME = 'RowCount'
THRESHOLD_VALUE = 'threshold_value'

# Column names in MDC Preprocessor
MDC_CHAT_HISTORY_COLUMN = 'chat_history'
MDC_CORRELATION_ID_COLUMN = 'correlationid'
MDC_DATA_COLUMN = 'data'
MDC_DATAREF_COLUMN = 'dataref'

# Column names in GenAI Preprocessor
GENAI_ROOT_SPAN_SCHEMA_COLUMN = "root_span"
GENAI_TRACE_ID_SCHEMA_COLUMN = "trace_id"

SCHEMA_INFER_ROW_COUNT = 10

AML_MOMO_ERROR_TAG = "azureml.modelmonitor.error"

TWO_SAMPLE_KOLMOGOROV_SMIRNOV_TEST_METRIC_NAME = "TwoSampleKolmogorovSmirnovTest"
PEARSONS_CHI_SQUARED_TEST_METRIC_NAME = "PearsonsChiSquaredTest"
JENSEN_SHANNON_DISTANCE_METRIC_NAME = "JensenShannonDistance"
POPULATION_STABILITY_INDEX_METRIC_NAME = "PopulationStabilityIndex"
NORMALIZED_WASSERSTEN_DISTANCE_METRIC_NAME = "NormalizedWassersteinDistance"

NULL_VALUE_RATE_METRIC_NAME = "NullValueRate"
DATA_TYPE_ERROR_RATE_METRIC_NAME = "DataTypeErrorRate"
OUT_OF_BOUNDS_RATE_METRIC_NAME = "OutOfBoundsRate"

NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN_METRIC_NAME = "NormalizedDiscountedCumulativeGain"

# gsq metric names
AGGREGATED_COHERENCE_PASS_RATE_METRIC_NAME = "AggregatedCoherencePassRate"
AGGREGATED_GROUNDEDNESS_PASS_RATE_METRIC_NAME = "AggregatedGroundednessPassRate"
AGGREGATED_FLUENCY_PASS_RATE_METRIC_NAME = "AggregatedFluencyPassRate"
AGGREGATED_SIMILARITY_PASS_RATE_METRIC_NAME = "AggregatedSimilarityPassRate"
AGGREGATED_RELEVANCE_PASS_RATE_METRIC_NAME = "AggregatedRelevancePassRate"

# model performance metric names
ACCURACY_METRIC_NAME = "Accuracy"
PERCISION_METRIC_NAME = "Precision"
RECALL_METRIC_NAME = "Recall"
MEAN_ABSOLUTE_ERROR_METRIC_NAME = "MeanAbsoluteError"
ROOT_MEAN_SQUARED_ERROR_METRIC_NAME = "RootMeanSquaredError"

# action analyzer
PROMPT_COLUMN = "prompt"
COMPLETION_COLUMN = "completion"
CONTEXT_COLUMN = "context"
TRACE_ID_COLUMN = "trace_id"
SPAN_ID_COLUMN = "span_id"
ROOT_QUESTION_COLUMN = "root_question"
TOPIC_LIST_COLUMN = "topic_list"
GROUP_LIST_COLUMN = "group_list"
VIOLATED_METRICS_COLUMN = "violated_metrics"
INDEX_CONTENT_COLUMN = "index_content"
INDEX_SCORE_COLUMN = "index_score"
INDEX_SCORE_LLM_COLUMN = "index_score_llm"
INDEX_ID_COLUMN = "index_id"
ROOT_SPAN_COLUMN = "root_span"
BAD_GROUP_COLUMN = "bad_group"
GOOD_GROUP_COLUMN = "good_group"
CONFIDENCE_SCORE_COLUMN = "confidence_score"
ACTION_ID_COLUMN = "action_id"
RETRIEVAL_QUERY_TYPE_COLUMN = "retrieval_query_type"
RETRIEVAL_TOP_K_COLUMN = "retrieval_top_k"
DEFAULT_TOPIC_NAME = "disparate"
PROMPT_FLOW_INPUT_COLUMN = "prompt_flow_input"

GSQ_METRICS_LIST = ["Fluency", "Coherence", "Groundedness", "Relevance", "Similarity"]
GOOD_METRICS_VALUE = 5
METRICS_VIOLATION_THRESHOLD = 4
RETRIEVAL_SPAN_TYPE = "Retrieval"
EMBEDDING_SPAN_TYPE = "Embedding"
TEXT_SPLITTER = "#<Splitter>#"

GROUP_TOPIC_MIN_SAMPLE_SIZE = 10
P_VALUE_THRESHOLD = 0.05
MEAN_THRESHOLD = 3

INDEX_ACTION_TYPE = "Index Action"
ACTION_DESCRIPTION = "The application's response quality is low due to suboptimal index retrieval. Please update the index with ID '{index_id}' to address this issue."  # noqa
MAX_SAMPLE_SIZE = 20
DEFAULT_RETRIEVAL_SCORE = 0

APP_TRACES_EVENT_LOG_INPUT_KEY = "promptflow.function.inputs"
APP_TRACES_EVENT_LOG_OUTPUT_KEY = "promptflow.function.output"
APP_TRACES_EVENT_LOG_RETRIEVAL_QUERY_KEY = "promptflow.retrieval.query"
APP_TRACES_EVENT_LOG_RETRIEVAL_DOCUMENT_KEY = "promptflow.retrieval.documents"
APP_TRACES_EVENT_LOG_EMBEDDINGS_KEY = "promptflow.embedding.embeddings"

# util
MLFLOW_RUN_ID = "MLFLOW_RUN_ID"
MAX_RETRY_COUNT = 3
