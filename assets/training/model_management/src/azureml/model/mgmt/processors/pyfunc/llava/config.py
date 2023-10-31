# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for LLaVA."""

from enum import Enum


MAX_PROMPT_LENGTH = 800


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class Tasks(_CustomEnum):
    """Task types supported by LLaVA."""

    IMAGE_TEXT_TO_TEXT = "image-text-to-text"


class MLflowLiterals:
    """MLflow export related literals."""

    MODEL_DIR = "model_dir"


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    INPUT_COLUMN_PROMPT = "prompt"
    INPUT_COLUMN_DIRECT_QUESTION = "direct_question"

    OUTPUT_COLUMN_RESPONSE = "response"
