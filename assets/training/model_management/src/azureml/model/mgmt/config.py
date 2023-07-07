# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model Management Config."""

from enum import Enum


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ModelType(_CustomEnum):
    """Enum for the Model Types accepted in ModelConfig."""

    MLFLOW = "mlflow_model"
    CUSTOM = "custom_model"


class ModelFlavor(_CustomEnum):
    """Enum for the Flavors accepted in ModelConfig."""

    TRANSFORMERS = "transformers"
    MMLAB_PYFUNC = "mmlab_pyfunc"


class PathType(_CustomEnum):
    """Enum for path types supported for model download."""

    AZUREBLOB = "AzureBlob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "GIT"  # Model hosted on a public GIT repo that can be cloned by GIT LFS.


class AppName:
    """Component AppName."""

    IMPORT_MODEL = "import_model"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL_TO_MLFLOW = "convert_model_to_mlflow"


class LoggerConfig:
    """Logger Config."""

    CODEC = "base64"
    INSTRUMENTATION_KEY = b"NzFiOTU0YTgtNmI3ZC00M2Y1LTk4NmMtM2QzYTY2MDVkODAz"
    MODEL_IMPORT_HANDLER_NAME = "ModelImportHandler"
    APPINSIGHT_HANDLER_NAME = "AppInsightsHandler"
    LOGGER_NAME = "FM_IMPORT_MODEL"
    VERBOSITY_LEVEL = "DEBUG"
    OFFLINE_RUN_MESSAGE = "OFFLINE_RUN"
    IMPORT_MODEL_VERSION = "0.0.8"  # Update when changing version in spec file.
