# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for text to image task."""

from enum import Enum
from mlflow.types import DataType


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Task types supported by stable diffusion."""

    TEXT_TO_IMAGE = "text-to-image"
    TEXT_TO_IMAGE_INPAINTING = "text-to-image-inpainting"


class DatatypeLiterals:
    """Literals related to data type."""

    IMAGE_FORMAT = "PNG"
    STR_ENCODING = "utf-8"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"
    MODEL_NAME = "model_name"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_PROMPT_DATA_TYPE = DataType.string
    INPUT_COLUMN_PROMPT = "prompt"
    OUTPUT_COLUMN_IMAGE_TYPE = DataType.binary
    OUTPUT_COLUMN_IMAGE = "generated_image"
    OUTPUT_COLUMN_NSFW_FLAG_TYPE = DataType.boolean
    OUTPUT_COLUMN_NSFW_FLAG = "nsfw_content_detected"

    # Extra constants for inpainting task
    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_IMAGE_TYPE = DataType.binary
    INPUT_COLUMN_MASK_IMAGE = "mask_image"
    INPUT_COLUMN_MASK_IMAGE_TYPE = DataType.binary


class BatchConstants:
    """Constants related to Batch inference."""

    BATCH_OUTPUT_PATH = "AZUREML_BI_OUTPUT_PATH"
