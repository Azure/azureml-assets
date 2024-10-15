# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Constants."""


class TelemetryConstants:
    """Telemetry Constants."""

    COMPONENT_NAME = "online_model_eval"
    COMPONENT_DEFAULT_VERSION = "0.0.1"

    EVALUATION_PREPROCESSOR_NAME = "evaluation_preprocessor"
    EVALUATION_POSTPROCESSOR_NAME = "evaluation_postprocessor"
    EVALUATION_EVALUATOR_NAME = "evaluation_evaluator"

    MODEL_EVALUATION_HANDLER_NAME = "AOAIModelEvaluationHandler"

    LOGGER_NAME = "online_model_evaluation_component"
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"

    NON_PII_MESSAGE = '[Hidden as it may contain PII]'


class ExceptionTypes:
    """AzureML Exception Types."""

    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}


class ExceptionLiterals:
    """Exception Constants."""

    MODEL_EVALUATION_TARGET = "AzureML Model Evaluation"
    DATA_TARGET = "AzureML Model Evaluation Data Validation"
    DATA_LOADING_TARGET = "AzureML Model Evaluation Data Loading"
    ARGS_TARGET = "AzureML Model Evaluation Arguments Validation"
    DATA_SAVING_TARGET = "AOAI Evaluation Data Saving"
    TemplateStrLoadingError = "AOAI EValuation: Failed to load template string"
    InferencingParsingException = "AOAI Evaluation: Failed to inference/parse from deployment"
    AuthenticationException = "Online Evaluation: Authentication failed"


class LogActivityLiterals:
    """Log Activity Constants."""

    LOG_AND_SAVE_OUTPUT = "log_and_save_output"
    DATA_LOADING = "loading_data"


class ErrorStrings:
    """Error Strings."""

    GenericOnlineEvalError = "Online Evaluation failed due to [{error}]"
    OnlineEvalAuthError = "Online Evaluation failed due to authentication error"
    OnlineEvalQueryError = "Please ensure to project trace_id and span_id from the query."
    BadInputData = "Failed to load data/config with error: [{error}]"
    SavingOutputError = "Failed to save output with error: [{error}]"
    DataNotFound = "No data found for the given query from the provided resource."
