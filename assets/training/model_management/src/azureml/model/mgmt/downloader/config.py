# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Downloaded Config."""

from enum import Enum


MLFLOW_MODEL = "mlflow_model"
MLMODEL = "MLmodel"
MLFLOW_MODEL_FOLDER = "mlflow_model_folder"


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class ModelSource(_CustomEnum):
    """Enum for supported model container sources."""

    HUGGING_FACE = "Huggingface"  # Model files hosted on Huggingface
    AZUREBLOB = "AzureBlob"  # Model files hosted on an AZUREBLOB blobstore with public read access.
    GIT = "GIT"  # Model hosted on a public GIT repo that can be cloned by GIT LFS.
