"""File containing constants for model evaluation script."""

MLFLOW_MODEL_TYPE_MAP = {
    "tabular-classification": "classifier",
    "text-classification": "text-classifier",
    "tabular-classification-multilabel": "classifier-multilabel",
    "text-classification-multilabel": "classifier-multilabel",
    "tabular-regression": "regressor",
    "text-named-entity-recognition": "text-ner",
    "forecasting": "forecasting",
    "text-translation": "translation",
    "question-answering": "question-answering",
    "text-summarization": "summarization"
}

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
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    REGRESSION = "tabular-regression"
    NER = "text-named-entity-recognition"
    FORECASTING = "forecasting"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"


ALL_TASKS = [
    TASK.CLASSIFICATION,
    TASK.CLASSIFICATION_MULTILABEL,
    TASK.REGRESSION,
    TASK.TEXT_CLASSIFICATION,
    TASK.TEXT_CLASSIFICATION_MULTILABEL,
    TASK.NER,
    TASK.FORECASTING,
    TASK.SUMMARIZATION,
    TASK.TRANSLATION,
    TASK.QnA]

CLASSIFICATION_SET = [TASK.CLASSIFICATION, TASK.CLASSIFICATION_MULTILABEL, TASK.TEXT_CLASSIFICATION, TASK.TEXT_CLASSIFICATION_MULTILABEL]
MULTIPLE_OUTPUTS_SET = [TASK.CLASSIFICATION_MULTILABEL, TASK.NER, TASK.TEXT_CLASSIFICATION_MULTILABEL]


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
