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

    TEXT_CLASSIFICATION = 'text-classification'
    TEXT_CLASSIFICATION_MULTILABEL = 'text-classification-multilabel'
    TEXT_NER = 'text-ner'


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_TEXT_DATA_TYPE = DataType.string
    INPUT_COLUMN_TEXT = "ReviewText"
