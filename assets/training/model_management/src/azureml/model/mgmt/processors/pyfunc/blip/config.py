# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common Config."""

from enum import Enum

from mlflow.types import DataType


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Tasks supported."""

    IMAGE_TO_TEXT = "image-to-text"


class HfBlipModelId(_CustomEnum):
    """Models supported."""

    BLIP_IMAGE_TO_TEXT = "Salesforce/blip-image-captioning-base"
    BLIP2_IMAGE_TO_TEXT = "Salesforce/blip2-opt-2.7b"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_IMAGE = "image"
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_TEXT = "text"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"
