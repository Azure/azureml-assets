# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains constants for data drift command components."""

# Column Names of Computed Metrics Parquet
FEATURE_COLUMN = 'feature_name'
FEATURE_CATEGORY_COLUMN = 'data_type'
METRIC_NAME_COLUMN = 'metric_name'
METRIC_VALUE_COLUMN = 'metric_value'

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

# Values for Outputs of Output Metrics Component
METADATA_VERSION = '1.0.0'

# Model types
CLASSIFICATION = "classification"
REGRESSION = "regression"
