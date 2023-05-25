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
COMPONENT_NAME_GENERATION_SAFETY_QUALITY_SIGNAL_MONITOR = (
    "generation_safety_quality_signal_monitor"
)

# MDC-generated target dataset of an iris model. The data contains drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_WITH_DRIFT = "azureml:iris_model_inputs_with_drift:1"


# MDC-generated target dataset of an iris model. The data contains no drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_NO_DRIFT = "azureml:iris_model_inputs_no_drift:1"


# MDC-generated target dataset of an iris model. The data contains no drift.
# Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_OUTPUTS_NO_DRIFT = "azureml:iris_model_outputs_no_drift:1"

# Iris baseline dataset as a MLTable.
DATA_ASSET_IRIS_BASELINE_DATA = "azureml:iris_baseline:1"

# Iris preprocessed target with dataset as a MLTable.
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_NO_DRIFT = (
    "azureml:iris_preprocessed_model_inputs_no_drift:1"
)
DATA_ASSET_IRIS_PREPROCESSED_MODEL_OUTPUTS_NO_DRIFT = (
    "azureml:iris_preprocessed_model_outputs_no_drift:1"
)
DATA_ASSET_IRIS_PREPROCESSED_MODEL_INPUTS_OUTPUTS_NO_DRIFT = (
    "azureml:iris_preprocessed_model_inputs_outputs_no_drift:1"
)

# MDC-generated target dataset of an iris model which contains both the input features as well as the inferred results.
# The data contains no drift. Output logs have been generated for 2023/01/01/00 and 2023/02/01/00.
DATA_ASSET_IRIS_MODEL_INPUTS_OUTPUTS_WITH_NO_DRIFT = (
    "azureml:iris_model_inputs_outputs_no_drift:1"
)

# Groundedness target dataset as a MLTable.
DATA_ASSET_GROUNDEDNESS_PREPROCESSED_TARGET_DATA = (
    "azureml:groundedness_preprocessed_target_small:1"
)
