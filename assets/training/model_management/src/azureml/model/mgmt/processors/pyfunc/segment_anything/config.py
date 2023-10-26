# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for SAM."""

from enum import Enum
from mlflow.types import DataType


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Task types supported by SAM."""

    MASK_GENERATION = "mask-generation"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_INPUT_POINTS = "input_points"
    INPUT_COLUMN_INPUT_BOXES = "input_boxes"
    INPUT_COLUMN_INPUT_LABELS = "input_labels"
    INPUT_COLUMN_IMAGE_DATA_TYPE = DataType.string
    INPUT_COLUMN_INPUT_POINTS_DATA_TYPE = DataType.string
    INPUT_COLUMN_INPUT_BOXES_DATA_TYPE = DataType.string
    INPUT_COLUMN_INPUT_LABELS_DATA_TYPE = DataType.string
    INPUT_PARAM_MULTIMASK_OUTPUT = "multimask_output"
    INPUT_PARAM_MULTIMASK_OUTPUT_DATA_TYPE = DataType.boolean
    OUTPUT_COLUMN_RESPONSE = "response"
    OUTPUT_COLUMN_DATA_TYPE = DataType.string
    RESPONSE_DF_PREDICTIONS = "predictions"
    RESPONSE_DF_MASKS_PER_PREDICTION = "masks_per_prediction"
    RESPONSE_DF_ENCODED_BINARY_MASK = "encoded_binary_mask"
    RESPONSE_DF_IOU_SCORE = "iou_score"


class SAMHFLiterals:
    """SAM HF related literals."""

    ORIGINAL_SIZES = "original_sizes"
    RESHAPE_INPUT_SIZES = "reshaped_input_sizes"


class DatatypeLiterals:
    """Literals related to data type."""

    IMAGE_FORMAT = "PNG"
