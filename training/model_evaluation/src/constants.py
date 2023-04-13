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


class TASK:
    """TASK list."""

    CLASSIFICATION = "tabular-classification"
    CLASSIFICATION_MULTILABEL = "tabular-classification-multilabel"
    REGRESSION = "tabular-regression"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    FORECASTING = "forecasting"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION = "text-generation"
    FILL_MASK = "fill-mask"
    FORECASTING = "forecasting"


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
    TASK.TEXT_GENERATION
]

CLASSIFICATION_SET = [
    TASK.CLASSIFICATION,
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL
]
MULTIPLE_OUTPUTS_SET = [TASK.CLASSIFICATION_MULTILABEL, TASK.NER, TASK.TEXT_CLASSIFICATION_MULTILABEL]

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
}


class TelemetryConstants:
    """Telemetry Constants."""

    COMPONENT_NAME = "model_evaluation"

    VALIDATION_NAME = "argument_validation"
    DATA_LOADING = "loading_data"

    ENVIRONMENT_SETUP = "environment_setup"

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

    INSTRUMENTATION_KEY = "71b954a8-6b7d-43f5-986c-3d3a6605d803"   # Vienna SDK


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


class ForecastingConfigContract:
    """Forecasting data contract on forecasting metrics config."""

    TIME_COLUMN_NAME = 'time_column_name'
    TIME_SERIES_ID_COLUMN_NAMES = 'time_series_id_column_names'
