# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common Config for MLflow model wrappers."""

from enum import Enum


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Tasks supported."""

    EMBEDDINGS = "embeddings"
    IMAGE_FEATURE_EXTRACTION = "image_feature_extraction"
    FEATURE_EXTRACTION = "feature_extraction"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    # For image-based tasks.
    INPUT_COLUMN_IMAGE = "image"
    OUTPUT_COLUMN_IMAGE_FEATURES = "image_features"
    # For text-based tasks.
    INPUT_COLUMN_TEXT = "text"
    OUTPUT_COLUMN_TEXT_FEATURES = "text_features"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"
    CHECKPOINT_PATH = "checkpoint_path"
    CONFIG_PATH = "config_path"
    DEVICE_TYPE = "device_type"
    TO_HALF_PRECISION = "to_half_precision"
