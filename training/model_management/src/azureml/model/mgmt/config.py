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
    TRITON = 'triton_model'


class ModelFlavor(_CustomEnum):
    """Enum for the Flavors accepted in ModelConfig."""

    HFTRANSFORMERS = 'hftransformers'
    PYTORCH = 'pytorch'


class PathType(_CustomEnum):
    """Enum for path types supported for model publishing."""

    AZUREBLOB = "azureblob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "git"      # Model hosted on a public GIT repo and can be cloned by GIT LFS.
