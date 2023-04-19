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


class PathType(_CustomEnum):
    """Enum for path types supported for model download."""

    AZUREBLOB = "AzureBlob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "GIT"  # Model hosted on a public GIT repo that can be cloned by GIT LFS.
