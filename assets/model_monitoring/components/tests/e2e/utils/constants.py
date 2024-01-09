# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains constants used by the e2e test suite."""

COMPONENT_NAME_DATA_DRIFT_SIGNAL_MONITOR = "data_drift_signal_monitor"
COMPONENT_NAME_PREDICTION_DRIFT_SIGNAL_MONITOR = "prediction_drift_signal_monitor"
COMPONENT_NAME_DATA_QUALITY_SIGNAL_MONITOR = "data_quality_signal_monitor"
COMPONENT_NAME_FEATURE_ATTRIBUTION_DRIFT_SIGNAL_MONITOR = (
    "feature_attribution_drift_signal_monitor"
)
COMPONENT_NAME_CREATE_MANIFEST = "model_monitor_create_manifest"
COMPONENT_NAME_MDC_PREPROCESSOR = "model_data_collector_preprocessor"
COMPONENT_NAME_METRIC_OUTPUTTER = "model_monitor_metric_outputter"
COMPONENT_NAME_DATA_JOINER = "model_monitor_data_joiner"
COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR = 'generation_safety_quality_signal_monitor'
COMPONENT_NAME_MODEL_PERFORMANCE_SIGNAL_MONITOR = "model_performance_signal_monitor"

# MDC-generated target dataset of an iris model. The data contains drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT = "azureml:uri_folder_iris_model_inputs_with_drift:1"
# TODO generate this data asset in the gated test workspace
DATA_ASSET_LLM_INPUTS = "azureml:ChatHistoryModelInput:1"


# MDC-generated target dataset of an iris model. The data contains no drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_NO_DRIFT = "azureml:uri_folder_iris_model_inputs_no_drift:1"


# MDC-generated target dataset of an iris model. The data contains no drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_OUTPUTS_NO_DRIFT = "azureml:uri_folder_iris_model_outputs_no_drift:1"

# Iris baseline dataset as a MLTable.
DATA_ASSET_IRIS_BASELINE_DATA = "azureml:mltable_iris_baseline:1"

# Iris baseline integer dataset as a MLTable.
DATA_ASSET_IRIS_BASELINE_INT_DATA_TYPE = "azureml:mltable_iris_baseline_int_data_type:1"

# Iris preprocessed target with integer dataset as a MLTable.
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT_INT_DATA = (
    "azureml:mltable_iris_preprocessed_model_inputs_no_drift_int_data:1"
)

# Iris preprocessed target with dataset as a MLTable.
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT = (
    "azureml:mltable_iris_preprocessed_model_inputs_no_drift:1"
)
DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_NO_DRIFT = (
    "azureml:mltable_iris_preprocessed_model_outputs_no_drift:1"
)
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_OUTPUTS_NO_DRIFT = (
    "azureml:mltable_iris_preprocessed_model_inputs_outputs_no_drift:1"
)

DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_WITH_JOIN_COLUMN = (
    "azureml:mltable_iris_preprocessed_model_inputs_with_join_column:1"
)

DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_OVERLAPPING_JOIN_VALUE = (
    "azureml:mltable_iris_preprocessed_model_inputs_no_overlapping_join_value:1"
)

DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_WITH_JOIN_COLUMN = (
    "azureml:mltable_iris_preprocessed_model_outputs_with_join_column:1"
)

DATA_ASSET_MLTABLE_DATA_DRIFT_SIGNAL_OUTPUT = (
    "azureml:mltable_data_drift_signal_output:1"
)

DATA_ASSET_MLTABLE_SAMPLES_INDEX_OUTPUT = (
    "azureml:mltable_samples_index_output:1"
)

DATA_ASSET_EMPTY = (
    "azureml:mltable_empty:1"
)

DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_COMMON_COLUMNS = (
    "azureml:mltable_iris_preprocessed_model_inputs_no_common_columns:1"
)

# used for checking against histogram regressions where a numerical data-column has a single distinct value
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_INT_SINGLE_VALUE_HISTOGRAM = (
    "azureml:mltable_iris_preprocessed_model_inputs_int_single_value_histogram:1"
)
DATA_ASSET_IRIS_BASELINE_INT_SINGLE_VALUE_HISTOGRAM = (
    "azureml:mltable_iris_baseline_int_single_value_histogram:1"
)

# MDC-generated target dataset of an iris model which contains both the input features as well as the inferred results.
# The data contains no drift. Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_OUTPUTS_WITH_NO_DRIFT = (
    "azureml:uri_folder_iris_model_inputs_outputs_no_drift:1"
)

DATA_ASSET_MODEL_INPUTS_JOIN_COLUMN_NAME = 'model_inputs_join_column'
DATA_ASSET_MODEL_OUTPUTS_JOIN_COLUMN_NAME = 'model_outputs_join_column'
# Groundedness target dataset as a MLTable.
DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA = 'azureml:groundedness_preprocess_target_small:1'

# For Data Quality with timestamp and boolean type in the MLTable
DATA_ASSET_VALID_DATATYPE = 'azureml:mltable_validate_datatype_for_data_quality:1'

DATA_ASSET_MODEL_PERFORMANCE_PRODUCTION_DATA = 'azureml:mltable_model_performance_production_data:1'
