# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logger Config."""


class AppName:
    """Component AppName."""

    IMPORT_MODEL = "import_model"
    REGISTER_MODEL = "register_model"
    DEPLOY_MODEL = "deploy_model"
    MLFLOW_MODEL_LOCAL_VALIDATION = "mlflow_model_local_validation"


class LoggerConfig:
    """Logger Config."""

    CODEC = "base64"
    INSTRUMENTATION_KEY = b"NzFiOTU0YTgtNmI3ZC00M2Y1LTk4NmMtM2QzYTY2MDVkODAz"
    MODEL_IMPORT_HANDLER_NAME = "ModelImportHandler"
    APPINSIGHT_HANDLER_NAME = "AppInsightsHandler"
    LOGGER_NAME = "FM_IMPORT_MODEL"
    VERBOSITY_LEVEL = "DEBUG"
    OFFLINE_RUN_MESSAGE = "OFFLINE_RUN"
    IMPORT_MODEL_VERSION = "0.0.7"  # Update when changing version in spec file.
