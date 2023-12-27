# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing constants for model evaluation script."""
from azureml.evaluate.mlflow.constants import ForecastFlavors

PREDICTIONS_COLUMN_NAME = "predictions"
TRANSFORMER_KEY = "y_transformer"
EVALUATION_RESULTS_PATH = "evaluationResult"

MLTABLE_FILE_NAME = "MLTable"
LLM_FT_PREPROCESS_FILENAME = "preprocess_args.json"
LLM_FT_TEST_DATA_KEY = "raw_test_data_fname"

RUN_PROPERTIES = {
    "showMetricsAtRoot": "true"
}

ROOT_RUN_PROPERTIES = {
    "PipelineType": "Evaluate"
}


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


class DEVICE:
    """Device list."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


ALL_DEVICES = [DEVICE.AUTO, DEVICE.CPU, DEVICE.GPU]


class MODEL_FLAVOR:
    """Model Flavors."""

    HFTRANSFORMERS = "hftransformers"
    HFTRANSFORMERSV2 = "hftransformersv2"
    TRANSFORMERS = "transformers"


ALL_MODEL_FLAVORS = [
    MODEL_FLAVOR.TRANSFORMERS,
    MODEL_FLAVOR.HFTRANSFORMERS,
    MODEL_FLAVOR.HFTRANSFORMERSV2
]


class TASK:
    """TASK list."""

    CLASSIFICATION = "tabular-classification"
    CLASSIFICATION_MULTILABEL = "tabular-classification-multilabel"
    REGRESSION = "tabular-regression"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION = "text-generation"
    FILL_MASK = "fill-mask"
    IMAGE_CLASSIFICATION = "image-classification"
    IMAGE_CLASSIFICATION_MULTILABEL = "image-classification-multilabel"
    IMAGE_OBJECT_DETECTION = "image-object-detection"
    IMAGE_INSTANCE_SEGMENTATION = "image-instance-segmentation"
    FORECASTING = "tabular-forecasting"
    CHAT_COMPLETION = "chat-completion"


class TRANSFORMERS_TASK:
    """Transformers Task list."""

    SUMMARIZATION = "summarization"


ALL_TASKS = [
    TASK.CLASSIFICATION,
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.REGRESSION,
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.NER,
    TASK.FORECASTING,
    TASK.SUMMARIZATION,
    TASK.QnA,
    TASK.TRANSLATION,
    TASK.FILL_MASK,
    TASK.TEXT_GENERATION,
    TASK.IMAGE_CLASSIFICATION,
    TASK.IMAGE_CLASSIFICATION_MULTILABEL,
    TASK.CHAT_COMPLETION,
    TASK.IMAGE_OBJECT_DETECTION,
    TASK.IMAGE_INSTANCE_SEGMENTATION,
]

MULTILABEL_SET = [
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.IMAGE_CLASSIFICATION_MULTILABEL
]

CLASSIFICATION_SET = [
    TASK.CLASSIFICATION,
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.IMAGE_CLASSIFICATION,
    TASK.IMAGE_CLASSIFICATION_MULTILABEL
]

MULTIPLE_OUTPUTS_SET = [
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.NER,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.IMAGE_CLASSIFICATION_MULTILABEL
]

MLFLOW_MODEL_TYPE_MAP = {
    TASK.CLASSIFICATION: "classifier",
    TASK.CLASSIFICATION_MULTILABEL: "classifier-multilabel",
    TASK.REGRESSION: "regressor",
    TASK.TEXT_CLASSIFICATION: "text-classifier",
    TASK.TEXT_CLASSIFICATION_MULTILABEL: "classifier-multilabel",
    TASK.NER: "text-ner",
    TASK.FORECASTING: "forecaster",
    TASK.TRANSLATION: "translation",
    TASK.QnA: "question-answering",
    TASK.SUMMARIZATION: "summarization",
    TASK.TEXT_GENERATION: "text-generation",
    TASK.FILL_MASK: "fill-mask",
    TASK.CHAT_COMPLETION: "chat-completion",
    TASK.IMAGE_CLASSIFICATION: "image-classifier",
    TASK.IMAGE_CLASSIFICATION_MULTILABEL: "image-classifier-multilabel",
    TASK.IMAGE_OBJECT_DETECTION: "image-object-detection",
    TASK.IMAGE_INSTANCE_SEGMENTATION: "image-instance-segmentation"
}

IMAGE_TASKS = [
    TASK.IMAGE_CLASSIFICATION,
    TASK.IMAGE_CLASSIFICATION_MULTILABEL,
    TASK.IMAGE_OBJECT_DETECTION,
    TASK.IMAGE_INSTANCE_SEGMENTATION
]

TEXT_TOKEN_TASKS = [
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.NER,
    TASK.TRANSLATION,
    TASK.QnA,
    TASK.SUMMARIZATION,
    TASK.TEXT_GENERATION,
    TASK.FILL_MASK,
    TASK.CHAT_COMPLETION,
]

TEXT_OUTPUT_TOKEN_TASKS = [
    TASK.TRANSLATION,
    TASK.QnA,
    TASK.SUMMARIZATION,
    TASK.TEXT_GENERATION,
    TASK.FILL_MASK,
    TASK.CHAT_COMPLETION,
]


class ChatCompletionConstants:
    """Chat completion constants."""

    OUTPUT = "predictions"
    OUTPUT_FULL_CONVERSATION = "prediction_appended"


class TelemetryConstants:
    """Telemetry Constants."""

    COMPONENT_NAME = "model_evaluation"
    COMPONENT_DEFAULT_VERSION = "0.0.21"

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


class ForecastingConfigContract:
    """Forecasting data contract on forecasting metrics config."""

    TIME_COLUMN_NAME = 'time_column_name'
    TIME_SERIES_ID_COLUMN_NAMES = 'time_series_id_column_names'
    FORECAST_FLAVOR = ForecastFlavors.FLAVOUR
    ROLLING_FORECAST_STEP = 'step'
    FORECAST_ORIGIN_COLUMN_NAME = 'forecast_origin_column_name'
    FORECAST_PREDICTIONS = "predictions_column_name"
    FORECAST_GROUND_TRUTH = "ground_truths_column_name"


class ForecastColumns:
    """The columns, returned in the forecast data frame."""

    _ACTUAL_COLUMN_NAME = '_automl_actual'
    _FORECAST_COLUMN_NAME = '_automl_forecast'
    _FORECAST_ORIGIN_COLUMN_DEFAULT = '_automl_forecast_origin'


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


ALLOWED_PIPELINE_HF_PARAMS = {
    "tokenizer_config",
    "generator_config",
    "model_kwargs",
    "pipeline_init_args",
    "trust_remote_code",
    "source_lang",
    "target_lang"
}

ALLOWED_PIPELINE_MLFLOW_TRANSFORMER_PARAMS = {
    "params"
}


class DataFrameParams:
    """DataFrame parameters for  dataset."""

    Ground_Truth_Column_Name = "ground_truths_column_name"
    Extra_Cols = "extra_cols"


class TextGenerationColumns:
    """Constants for Text Generation tasks."""

    SUBTASKKEY = "sub_task"
    Text_GEN_TEST_CASE_COLUMN_NAME = "test_case_column_name"
    Text_GEN_Y_TEST_COLUMN_NAME = "y_test_column_name"


class SubTask:
    """Constants for sub-tasks."""

    SUB_TASK_KEY = "sub_task"

    CODEGENERATION = "code"
    RAG_EVALUATION = "rag"


class OpenAIConstants:
    """OpenAI Related Constants."""

    CONNECTION_STRING_KEY = "AZUREML_WORKSPACE_CONNECTION_ID_AOAI"
    METRICS_KEY = "openai_params"
    DEFAULT_OPENAI_CONFIG = {
        "type": "azure_open_ai",
        "model_name": "gpt-35-turbo-16k",
        "deployment_name": "gpt-35-turbo-16k"
    }
    DEFAULT_OPENAI_INIT_PARAMS = {
        "openai_api_type": "azure",
        "openai_api_version": "2023-03-15-preview"
    }
    QUESTIONS_KEY = "questions"
    CONTEXTS_KEY = "contexts"
    REQUIRED_KEYS = [QUESTIONS_KEY, CONTEXTS_KEY]
