# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


COMPONENT_NAME = "azureml.rai.tabular"


class DashboardInfo:
    MODEL_ID_KEY = "id"  # To match Model schema
    MODEL_INFO_FILENAME = "model_info.json"
    TRAIN_FILES_DIR = "train"
    TEST_FILES_DIR = "test"

    RAI_INSIGHTS_MODEL_ID_KEY = "model_id"
    RAI_INSIGHTS_RUN_ID_KEY = "rai_insights_parent_run_id"
    RAI_INSIGHTS_GATHER_RUN_ID_KEY = "rai_insights_gather_run_id"
    RAI_INSIGHTS_CONSTRUCTOR_ARGS_KEY = "constructor_args"
    RAI_INSIGHTS_PARENT_FILENAME = "rai_insights.json"
    RAI_INSIGHTS_TRAIN_DATASET_ID_KEY = "train_dataset_id"
    RAI_INSIGHTS_TEST_DATASET_ID_KEY = "test_dataset_id"
    RAI_INSIGHTS_DASHBOARD_TITLE_KEY = "dashboard_title"
    RAI_INSIGHTS_INPUT_ARGS_KEY = "rai_insight_input_args"


class PropertyKeyValues:
    # The property to indicate the type of Run
    RAI_INSIGHTS_TYPE_KEY = "_azureml.responsibleai.rai_insights.type"
    RAI_INSIGHTS_TYPE_CONSTRUCT = "construction"
    RAI_INSIGHTS_TYPE_CAUSAL = "causal"
    RAI_INSIGHTS_TYPE_COUNTERFACTUAL = "counterfactual"
    RAI_INSIGHTS_TYPE_EXPLANATION = "explanation"
    RAI_INSIGHTS_TYPE_ERROR_ANALYSIS = "error_analysis"
    RAI_INSIGHTS_TYPE_GATHER = "gather"

    # Property to point at the model under examination
    RAI_INSIGHTS_MODEL_ID_KEY = "_azureml.responsibleai.rai_insights.model_id"

    # Property for tool runs to point at their constructor run
    RAI_INSIGHTS_CONSTRUCTOR_RUN_ID_KEY = (
        "_azureml.responsibleai.rai_insights.constructor_run"
    )

    # Property to record responsibleai version
    RAI_INSIGHTS_RESPONSIBLEAI_VERSION_KEY = (
        "_azureml.responsibleai.rai_insights.responsibleai_version"
    )

    # Property format to indicate presence of a tool
    RAI_INSIGHTS_TOOL_KEY_FORMAT = "_azureml.responsibleai.rai_insights.has_{0}"

    # Dashboard id
    RAI_INSIGHTS_DASHBOARD_ID_KEY = "_azureml.responsibleai.rai_insights.dashboard_id"
    # Dashboard title
    RAI_INSIGHTS_DASHBOARD_TITLE_KEY = (
        "_azureml.responsibleai.rai_insights.dashboard_title"
    )

    # RAI insight score card key
    RAI_INSIGHTS_SCORE_CARD_TITLE_KEY = (
        "_azureml.responsibleai.rai_insights.rai_scorecard_title"
    )

    # Dataset ids
    RAI_INSIGHTS_TRAIN_DATASET_ID_KEY = (
        "_azureml.responsibleai.rai_insights.train_dataset_id"
    )
    RAI_INSIGHTS_TEST_DATASET_ID_KEY = (
        "_azureml.responsibleai.rai_insights.test_dataset_id"
    )

    RAI_INSIGHTS_DROPPED_FEATURE_KEY = "dropped_features"
    RAI_INSIGHTS_IDENTITY_FEATURE_KEY = "identity_feature_name"
    RAI_INSIGHTS_DATETIME_FEATURES_KEY = "datetime_features"
    RAI_INSIGHTS_TIME_SERIES_ID_FEATURES_KEY = "time_series_id_features"

    RAI_INSIGHTS_DATA_TYPE_KEY = "data_type"


class RAIToolType:
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    ERROR_ANALYSIS = "error_analysis"
    EXPLANATION = "explanation"
    SCORECARD = "scorecard"


MLFLOW_MODEL_SERVER_PORT = 5432
