# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File containing constants for model evaluation script."""

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


class DEVICE:
    """Device list."""

    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


ALL_DEVICES = [DEVICE.AUTO, DEVICE.CPU, DEVICE.GPU]


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
    TASK.IMAGE_OBJECT_DETECTION,
    TASK.IMAGE_INSTANCE_SEGMENTATION
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
    TASK.FORECASTING: "forecasting",
    TASK.TRANSLATION: "translation",
    TASK.QnA: "question-answering",
    TASK.SUMMARIZATION: "summarization",
    TASK.TEXT_GENERATION: "text-generation",
    TASK.FILL_MASK: "fill-mask",
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


class TelemetryConstants:
    """Telemetry Constants."""

    COMPONENT_NAME = "model_evaluation"

    VALIDATION_NAME = "argument_validation"
    DATA_LOADING = "loading_data"
    LOG_AND_SAVE_OUTPUT = "log_and_save_output"

    LOAD_MODEL = "load_model"

    PREDICT_NAME = "predict"
    MODEL_PREDICTION_NAME = "model_prediction"
    COMPUTE_METRICS_NAME = "compute_metrics"
    SCORE_NAME = "score"
    EVALUATE_MODEL_NAME = "evaluate_model"

    MLFLOW_NAME = "mlflow_evaluate"

    MODEL_EVALUATION_HANDLER_NAME = "ModelEvaluationHandler"
    LOGGER_NAME = "model_evaluation_component"
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"


class ExceptionLiterals:
    """Exception Constants."""

    MODEL_EVALUATION_TARGET = "AzureML Model Evaluation"
    DATA_TARGET = "AzureML Model Evaluation Data Validation"
    DATA_LOADING_TARGET = "AzureML Model Evaluation Data Loading"
    ARGS_TARGET = "AzureML Model Evaluation Arguments Validation"
    MODEL_LOADER_TARGET = "AzureML Model Evaluation Model Loading"


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

    # Arguments related
    ArgumentParsingError = "Failed to parse input arguments."
    InvalidTaskType = "Given Task Type [{TaskName}] is not supported. " + \
                      "Please see the list of supported task types:\n" + \
                      "\n".join(ALL_TASKS)
    InvalidModel = "Either correct Model URI or Mlflow Model should be passed.\n" \
                   "If you have passed Model URI, your Model URI is incorrect."
    BadModelData = "Model load failed due to error: [{error}]"
    InvalidTestData = "Test data should be passed."
    InvalidPredictionsData = "Predictions should be passed."
    InvalidGroundTruthData = "Ground truth should be passed."
    InvalidGroundTruthColumnName = "Ground truth column name should be passed since columns in data are > 0."
    InvalidGroundTruthColumnNameData = "Ground truth column name not found in input data."
    InvalidPredictionColumnNameData = "Prediction Column name not found in input data."

    # Data Asset related
    BadLabelColumnName = "No label column found in test data."
    BadFeatureColumnNames = "input_column_names is not a subset of input test dataset columns.\
                 input_column_names include [{keep_columns}] whereas data has [{data_columns}]"
    BadInputData = "Failed to load data with error: [{error}]"
    BadEvaluationConfigFile = "Evaluation Config file failed to load due to [{error}]"
    BadEvaluationConfigParam = "Evaluation Config Params failed to load due to [{error}]"
    BadEvaluationConfig = "Evaluation Config failed to load due to [{error}]"

    BadForecastGroundTruthData = "For forecasting tasks, the table needs to be provided " \
                                 "in the ground_truths parameter." \
                                 "The table must contain time, prediction " \
                                 "ground truth and time series IDs columns."
    BadRegressionColumnType = "Expected target columns of type float found [{y_test_dtype}] instead"

    # Logging Related
    MetricLoggingError = "Failed to log metric {metric_name} due to [{error}]"


class ForecastingConfigContract:
    """Forecasting data contract on forecasting metrics config."""

    TIME_COLUMN_NAME = 'time_column_name'
    TIME_SERIES_ID_COLUMN_NAMES = 'time_series_id_column_names'
    FORECAST_ORIGIN_COLUMN_NAME = 'forecast_origin_column_name'


class ForecastColumns:
    """The columns, returned in the forecast data frame."""

    _ACTUAL_COLUMN_NAME = '_automl_actual'
    _FORECAST_COLUMN_NAME = '_automl_forecast'
    _FORECAST_ORIGIN_COLUMN_DEFAULT = '_automl_forecast_origin'


ALLOWED_PIPELINE_PARAMS = {
    "tokenizer_config",
    "model_kwargs",
    "pipeline_init_args",
    "trust_remote_code",
    "source_lang",
    "target_lang"
}
