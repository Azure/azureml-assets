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
