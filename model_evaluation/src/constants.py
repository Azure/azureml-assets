MLFLOW_MODEL_TYPE_MAP = {
    "classification":"classifier",
    "regression": "regressor", 
    "text-ner": "text-ner"
}

PREDICTIONS_COLUMN_NAME = "predictions"
TRANSFORMER_KEY = "y_transformer"
EVALUATION_RESULTS_PATH = "evaluationResult"

MLTABLE_FILE_NAME = "MLTable"

class TASK:
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    NER = "text-ner"
    FORECASTING = "forecasting"

ALL_TASKS = [TASK.CLASSIFICATION, TASK.REGRESSION, TASK.NER, TASK.FORECASTING]

class TelemetryConstants:
    COMPONENT_NAME = "model_evaluation"

    VALIDATION_NAME = "argument_validation"
    DATA_LOADING = "loading_data"

    ENVIRONMENT_SETUP = "environment_setup"

    PREDICT_NAME = "predict"
    COMPUTE_METRICS_NAME = "compute_metrics"
    SCORE_NAME = "score"

    MLFLOW_NAME = "mlflow_evaluate"

    MODEL_EVALUATION_HANDLER_NAME = "ModelEvaluationHandler"
    LOGGER_NAME = "model_evaluation_component"
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"

    INSTRUMENTATION_KEY = "71b954a8-6b7d-43f5-986c-3d3a6605d803" #Vienna SDK

class ExceptionLiterals:
    MODEL_EVALUATION_TARGET = "AzureML Model Evaluation"
    DATA_TARGET = "AzureML Model Evaluation Data Validation"
    DATA_LOADING_TARGET = "AzureML Model Evaluation Data Loading"
    ARGS_TARGET = "AzureML Model Evaluation Arguments Validation"
    MODEL_LOADER_TARGET = "AzureML Model Evaluation Model Loading"
    

class ExceptionTypes:
    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}