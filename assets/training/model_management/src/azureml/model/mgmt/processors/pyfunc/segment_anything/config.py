# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for LLaVA."""

from enum import Enum
from mlflow.types import DataType


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Task types supported by LLaVA."""

    SEGMENT_ANYTHING = "segment-anything"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_INPUT_POINTS = "input_points"
    INPUT_COLUMN_INPUT_BOXES = "input_boxes"
    INPUT_COLUMN_INPUT_LABELS = "input_labels"
    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.binary
    INPUT_COLUMN_INPUT_POINTS_DATA_TYPE = DataType.string
    INPUT_COLUMN_INPUT_BOXES_DATA_TYPE = DataType.string
    INPUT_COLUMN_INPUT_LABELS_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    OUTPUT_COLUMN_MASKS = "masks"
    OUTPUT_COLUMN_IOU_SCORES = "iou_scores"

class BatchConstants:
    """Constants related to Batch inference."""

    BATCH_OUTPUT_PATH = "AZUREML_BI_OUTPUT_PATH"

class DatatypeLiterals:
    """Literals related to data type."""

    IMAGE_FORMAT = "PNG"
    STR_ENCODING = "utf-8"