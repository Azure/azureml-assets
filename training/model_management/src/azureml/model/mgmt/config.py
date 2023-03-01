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

    MLFLOW = 'mlflow_model'
    CUSTOM = 'custom_model'


class PathType(_CustomEnum):
    """Enum for path types supported for model download."""

    AZUREBLOB = "AzureBlob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "GIT"              # Model hosted on a public GIT repo that can be cloned by GIT LFS.


class ModelSource(_CustomEnum):
    """Enum for supported model container sources."""

    HUGGING_FACE = "Huggingface"   # Model files hosted on Huggingface

    @classmethod
    def get_base_url(cls, key) -> str:
        """Return base url of the model container."""
        global MODEL_CONTAINER_DICT
        if isinstance(key, str):
            return MODEL_CONTAINER_DICT.get(key).get("base_uri")
        return MODEL_CONTAINER_DICT.get(key.value).get("base_uri")

    @classmethod
    def get_path_type(cls, key) -> PathType:
        """Return path type of the model container."""
        global MODEL_CONTAINER_DICT
        if isinstance(key, str):
            return MODEL_CONTAINER_DICT.get(key).get("path_type")
        return MODEL_CONTAINER_DICT.get(key.value).get("path_type")


MODEL_CONTAINER_DICT = {
    ModelSource.HUGGING_FACE.value: {
        "base_uri": "https://huggingface.co/{}",
        "path_type":  PathType.GIT
    }
}
