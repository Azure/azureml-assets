# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model Management Config."""

from enum import Enum


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list_values(cls):
        _dict = list(cls._value2member_map_.values())
        return [_enum.value for _enum in _dict]


class ModelType(_CustomEnum):
    """Enum for the Model Types accepted in ModelConfig."""

    MLFLOW = "mlflow_model"
    CUSTOM = "custom_model"


class ModelFramework(_CustomEnum):
    """Enum for the model framework accepted by model preprocess."""

    HUGGINGFACE = "Huggingface"
    MMLAB = "MMLab"
    LLAVA = "llava"
    AutoML = "AutoML"


class PathType(_CustomEnum):
    """Enum for path types supported for model download."""

    AZUREBLOB = "AzureBlob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "GIT"  # Model hosted on a public GIT repo that can be cloned by GIT LFS.


class AppName:
    """Component AppName."""

    IMPORT_MODEL = "import_model"
    DOWNLOAD_MODEL = "download_model"
    CONVERT_MODEL_TO_MLFLOW = "convert_model_to_mlflow"
    VALIDATION_TRIGGER_IMPORT = "validation_trigger_import"


class LoggerConfig:
    """Logger Config."""

    CODEC = "base64"
    INSTRUMENTATION_KEY = b"NzFiOTU0YTgtNmI3ZC00M2Y1LTk4NmMtM2QzYTY2MDVkODAz"
    MODEL_IMPORT_HANDLER_NAME = "ModelImportHandler"
    APPINSIGHT_HANDLER_NAME = "AppInsightsHandler"
    LOGGER_NAME = "FM_IMPORT_MODEL"
    VERBOSITY_LEVEL = "DEBUG"
    OFFLINE_RUN_MESSAGE = "OFFLINE_RUN"
    ASSET_NOT_FOUND = "AssetID missing in run details"


class LlamaHFModels(_CustomEnum):
    """Llama HF Models."""

    LLAMA_2_7B_CHAT_HF = "meta-llama/Llama-2-7b-chat-hf"
    LLAMA_2_13B_CHAT_HF = "meta-llama/Llama-2-13b-chat-hf"
    LLAMA_2_70B_CHAT_HF = "meta-llama/Llama-2-70b-chat-hf"
    LLAMA_2_7B_HF = "meta-llama/Llama-2-7b-hf"
    LLAMA_2_13B_HF = "meta-llama/Llama-2-13b-hf"
    LLAMA_2_70B_HF = "meta-llama/Llama-2-70b-hf"


class LlamaModels(_CustomEnum):
    """Llama Models."""

    LLAMA_2_7B_CHAT = "meta-llama/Llama-2-7b-chat"
    LLAMA_2_13B_CHAT = "meta-llama/Llama-2-13b-chat"
    LLAMA_2_70B_CHAT = "meta-llama/Llama-2-70b-chat"
    LLAMA_2_7B = "meta-llama/Llama-2-7b"
    LLAMA_2_13B = "meta-llama/Llama-2-13b"
    LLAMA_2_70B = "meta-llama/Llama-2-70b"


class LlamaModelsInRegistry(_CustomEnum):
    """Llama Models in Registry."""

    LLAMA_2_7B_CHAT = "Llama-2-7b-chat"
    LLAMA_2_13B_CHAT = "Llama-2-13b-chat"
    LLAMA_2_70B_CHAT = "Llama-2-70b-chat"
    LLAMA_2_7B = "Llama-2-7b"
    LLAMA_2_13B = "Llama-2-13b"
    LLAMA_2_70B = "Llama-2-70b"


llama_dict = {
    LlamaHFModels.LLAMA_2_13B_CHAT_HF.value: LlamaModelsInRegistry.LLAMA_2_13B_CHAT.value,
    LlamaHFModels.LLAMA_2_7B_CHAT_HF.value: LlamaModelsInRegistry.LLAMA_2_7B_CHAT.value,
    LlamaHFModels.LLAMA_2_70B_CHAT_HF.value: LlamaModelsInRegistry.LLAMA_2_70B_CHAT.value,
    LlamaHFModels.LLAMA_2_7B_HF.value: LlamaModelsInRegistry.LLAMA_2_7B.value,
    LlamaHFModels.LLAMA_2_70B_HF.value: LlamaModelsInRegistry.LLAMA_2_70B.value,
    LlamaHFModels.LLAMA_2_13B_HF.value: LlamaModelsInRegistry.LLAMA_2_13B.value,
    LlamaModels.LLAMA_2_7B_CHAT.value: LlamaModelsInRegistry.LLAMA_2_7B_CHAT.value,
    LlamaModels.LLAMA_2_70B_CHAT.value: LlamaModelsInRegistry.LLAMA_2_70B_CHAT.value,
    LlamaModels.LLAMA_2_7B.value: LlamaModelsInRegistry.LLAMA_2_7B.value,
    LlamaModels.LLAMA_2_70B.value: LlamaModelsInRegistry.LLAMA_2_70B.value,
    LlamaModels.LLAMA_2_13B.value: LlamaModelsInRegistry.LLAMA_2_13B.value,
    LlamaModels.LLAMA_2_13B_CHAT.value: LlamaModelsInRegistry.LLAMA_2_13B_CHAT.value,
}
