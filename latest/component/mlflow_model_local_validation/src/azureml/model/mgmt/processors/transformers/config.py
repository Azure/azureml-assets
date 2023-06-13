# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers Config."""

from enum import Enum
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelWithLMHead,
    AutoModelForMaskedLM,
    AutoModelForImageClassification,
    WhisperForConditionalGeneration,
)
from diffusers import StableDiffusionPipeline
from typing import Any


MLFLOW_ARTIFACT_DIRECTORY = "mlflow_model_folder"

# HF flavor patterns
MODEL_CONFIG_FILE_PATTERN = r"^config\.json$"
MODEL_FILE_PATTERN = r"^pytorch.*$"
TOKENIZER_FILE_PATTERN = r"^tokenizer.*|vocab\.json$"


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


class SupportedVisionTasks(_CustomEnum):
    """Supported Vision Hugging face tasks."""

    IMAGE_CLASSIFICATION = "image-classification"


class SupportedNLPTasks(_CustomEnum):
    """Supported NLP Hugging face tasks."""

    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TRANSLATION = "translation"


class SupportedTextToImageVariants(_CustomEnum):
    """Supported text to image variants."""

    STABLE_DIFFUSION = "stable-diffusion"


class SupportedASRVariants(_CustomEnum):
    """Supported text to image variants."""

    WHISPER_ASR = "whisper-asr"


class SupportedTasks(_CustomEnum):
    """Supported Hugging face tasks for conversion to mlflow."""

    # NLP tasks
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TRANSLATION = "translation"
    # Vision tasks
    IMAGE_CLASSIFICATION = "image-classification"
    # Text to Image
    TEXT_TO_IMAGE = "text-to-image"
    # ASR
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"


class TaskToClassMapping:
    """Mapping of supported hugging face tasks to respective AutoModel classes."""

    _task_loader_mapping = {
        "fill-mask": AutoModelForMaskedLM,
        "text-classification": AutoModelForSequenceClassification,
        "token-classification": AutoModelForTokenClassification,
        "question-answering": AutoModelForQuestionAnswering,
        "summarization": AutoModelWithLMHead,
        "text-generation": AutoModelWithLMHead,
        "translation": AutoModelForSeq2SeqLM,
        "image-classification": AutoModelForImageClassification,
        "stable-diffusion": StableDiffusionPipeline,
        "whisper-asr": WhisperForConditionalGeneration,
    }

    def get_loader_class(task_type) -> Any:
        """Return loader class for a supported Hugging face task."""
        return TaskToClassMapping._task_loader_mapping.get(task_type)

    def get_loader_class_name(task_type) -> str:
        """Return loader class name for a supported Hugging face task."""
        cls = TaskToClassMapping.get_loader_class(task_type)
        return cls.__name__ if cls else None
