# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa: E702

"""This module defines the EngineName and TaskType enums."""
import sys

sys.path.append("/src/")
from llm.optimized.inference.constants import ALL_TASKS

MLTABLE_FILE_NAME = "MLTable"
LLM_FT_PREPROCESS_FILENAME = "preprocess_args.json"
LLM_FT_TEST_DATA_KEY = "raw_test_data_fname"
LLM_FT_CHAT_COMPLETION_KEY = "messages"

# default values
class ModelPath:
    """Model Path related constants."""

    MLMODEL_PATH = "MLmodel"
    DEPRECATED_MLFLOW_MODEL_PATH = "data/model"
    DEPRECATED_MLFLOW_CONFIG_PATH = "data/config"
    DEPRECATED_MLFLOW_TOKENIZER_PATH = "data/tokenizer"
    DEFAULT_TOKENIZER_FILE = "tokenizer_config.json"
    DEFAULT_MLFLOW_MODEL_PATH = "model"
    DEFAULT_TOKENIZER_PATH = "components/tokenizer"


class ArgumentLiterals:
    """Input Argument literals list."""

    TASK = "task"
    DATA = "data"
    MLFLOW_MODEL = "mlflow_model"
    LABEL_COLUMN_NAME = "label_column_name"
    INPUT_COLUMN_NAMES = "input_column_names"
    DEVICE = "device"
    BATCH_SIZE = "batch_size"
    CONFIG_FILE_NAME = "config_file_name"
    CONFIG_STR = "config_str"
    CONFIG = "config"

    GROUND_TRUTHS = "ground_truths"
    GROUND_TRUTHS_COLUMN_NAME = "ground_truths_column_name"
    PREDICTIONS = "predictions"
    PREDICTIONS_COLUMN_NAME = "predictions_column_name"
    PREDICTION_PROBABILITIES = "prediction_probabilities"
    PERFORMANCE_METADATA = "performance_metadata"
    OUTPUT = "output"
    OPENAI_CONFIG_PARAMS = "openai_config_params"
    PRS_DATA = "prs_data"

class ErrorStrings:
    """Error Strings."""

    GenericModelEvaluationError = "Model Evaluation failed due to [{error}]"
    GenericModelPredictionError = "Model Prediction failed due to [{error}]"
    GenericComputeMetricsError = "Compute metrics failed due to [{error}]"

    # Download dependencies
    DownloadDependenciesFailed = "Failed to install model dependencies: [{dependencies}]"

    # Arguments related
    ArgumentParsingError = "Parsing input arguments failed with error: [{error}]"
    InvalidTaskType = "Given Task Type [{TaskName}] is not supported. " + \
                      "Please see the list of supported task types:\n" + \
                      "\n".join(ALL_TASKS)
    InvalidModel = "Model passed is not a valid MLFlow model. " + \
                   "Please save model using 'azureml-evaluate-mlflow' or 'mlflow' package."
    BadModelData = "Model load failed due to error: [{error}]"
    InvalidData = "[{input_port}] should be passed."
    InvalidFileInputSource = "File input source [{input_port}] must be of type ro_mount."
    InvalidGroundTruthColumnName = "Ground truth column name should be passed since columns in data are > 0."
    InvalidGroundTruthColumnNameData = "Ground truth column name not found in input data."
    InvalidPredictionColumnNameData = "Prediction Column name [{prediction_column}] not found in input data."
    InvalidYTestCasesColumnNameData = "y_test_cases column name not found in input data."
    InvalidGroundTruthColumnNameCodeGen = "The format for the label column name in code generation should follow " \
                                          "the pattern: '<label_col_name>,<test_case_col_name>'. Either " \
                                          "<label_col_name> or <test_case_col_name> can be empty, but at least one " \
                                          "of them must be set."

    # Data Asset related
    BadInputColumnData = "No input columns found in test data."
    BadLabelColumnName = "No label column found in test data."
    BadFeatureColumnNames = "[{column}] not a subset of input test dataset columns.\
                 [{column}] include [{keep_columns}] whereas data has [{data_columns}]"
    BadInputData = "Failed to load data with error: [{error}]"
    EmptyInputData = "Input data contains no data."
    BadEvaluationConfigFile = "Evaluation Config file failed to load due to [{error}]"
    BadEvaluationConfigParam = "Evaluation Config Params failed to load due to [{error}]"
    BadEvaluationConfig = "Evaluation Config failed to load due to [{error}]"

    BadForecastGroundTruthData = "For forecasting tasks, the table needs to be provided " \
                                 "in the ground_truths parameter." \
                                 "The table must contain time, prediction " \
                                 "ground truth and time series IDs columns."
    BadRegressionColumnType = "Failed to convert y_test column type to float with error: [{error}]. " \
                              "Expected target columns of type float found [{y_test_dtype}] instead"
    FilteringDataError = "Failed to filter data with error: [{error}]"

    # Logging Related
    MetricLoggingError = "Failed to log metric {metric_name} due to [{error}]"
    SavingOutputError = "Failed to save output due to [{error}]"

class TelemetryConstants:
    """Telemetry Constants."""

    COMPONENT_NAME = "model_evaluation"
    COMPONENT_DEFAULT_VERSION = "0.0.17"

    INITIALISING_RUNNER = "initialising_runner"
    VALIDATION_NAME = "argument_validation"
    DATA_LOADING = "loading_data"
    LOG_AND_SAVE_OUTPUT = "log_and_save_output"

    LOAD_MODEL = "load_model"

    PREDICT_NAME = "predict"
    TRIGGER_VALIDATION_NAME = "validation_trigger_model_evaluation"
    MODEL_PREDICTION_NAME = "model_prediction"
    COMPUTE_METRICS_NAME = "compute_metrics"
    SCORE_NAME = "score"
    EVALUATE_MODEL_NAME = "evaluate_model"
    DOWNLOAD_MODEL_DEPENDENCIES = "download_model_dependencies"

    MLFLOW_NAME = "mlflow_evaluate"

    MODEL_EVALUATION_HANDLER_NAME = "ModelEvaluationHandler"
    LOGGER_NAME = "model_evaluation_component"
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"

    NON_PII_MESSAGE = '[Hidden as it may contain PII]'


class ExceptionLiterals:
    """Exception Constants."""

    MODEL_EVALUATION_TARGET = "AzureML Model Evaluation"
    DATA_TARGET = "AzureML Model Evaluation Data Validation"
    DATA_LOADING_TARGET = "AzureML Model Evaluation Data Loading"
    ARGS_TARGET = "AzureML Model Evaluation Arguments Validation"
    MODEL_LOADER_TARGET = "AzureML Model Evaluation Model Loading"
    DATA_SAVING_TARGET = "AzureML Model Evaluation Data Saving"


class ExceptionTypes:
    """AzureML Exception Types."""

    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}

class PerformanceColumns:
    """The column names for the performance metadata output."""

    BATCH_SIZE_COLUMN_NAME = 'batch_size'
    START_TIME_COLUMN_NAME = 'start_time_iso'
    END_TIME_COLUMN_NAME = 'end_time_iso'
    LATENCY_COLUMN_NAME = 'time_taken_ms'
    INPUT_CHARACTERS_COLUMN_NAME = 'input_character_count'
    OUTPUT_CHARACTERS_COLUMN_NAME = 'output_character_count'
    INPUT_TOKENS_COLUMN_NAME = 'input_token_count'
    OUTPUT_TOKENS_COLUMN_NAME = 'output_token_count'

class TASK:
    """TASK list."""

    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION = "text-generation"
    TEXT_GENERATION_CODE = "text-generation-code"
    FILL_MASK = "fill-mask"
    CHAT_COMPLETION = "chat-completion"


TEXT_TOKEN_TASKS = [
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.NER,
    TASK.TRANSLATION,
    TASK.QnA,
    TASK.SUMMARIZATION,
    TASK.TEXT_GENERATION,
    TASK.FILL_MASK,
    TASK.CHAT_COMPLETION
]

FILTER_MODEL_PREDICTION_PARAMS = [
    "tokenizer_config",
    "generator_config"
]