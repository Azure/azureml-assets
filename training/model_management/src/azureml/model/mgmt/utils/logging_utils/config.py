# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Config file used for logging
"""


class APPName:
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL_TO_MLFLOW = "convert_model_to_mlflow"
    MLFLOW_MODEL_LOCAL_VALIDATION = "mlflow_model_local_validation"


class Config:
    """
    config class
    """

    VERBOSITY_LEVEL = "DEBUG"
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"
    AMLFT_HANDLER_NAME = "AmlFtHandlerName"
    LOGGER_NAME = "generic_finetune_component"

    INSTRUMENTATION_KEY_AML = "7b709447-0334-471a-9648-30349a41b45c"
    INSTRUMENTATION_KEY_AML_OLD = "71b954a8-6b7d-43f5-986c-3d3a6605d803"
    OFFLINE_RUN_MESSAGE = "This is an offline run"
