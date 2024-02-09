# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""HFTransformers Config."""

from enum import Enum


MLFLOW_ARTIFACT_DIRECTORY = "mlflow_model_folder"

# HF flavor patterns
MODEL_CONFIG_FILE_PATTERN = r"^config\.json|.+\.py$"
MODEL_FILE_PATTERN = r"^pytorch.*|.+\.py$"
TOKENIZER_FILE_PATTERN = r"^.*token.*|.*vocab.*|.*processor.*|.+\.py$"
META_FILE_PATTERN = r"^.*README.*|.*LICENSE.*|.*USE_POLICY.*$"


class _CustomEnum(Enum):
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def list_values(cls):
        _dict = list(cls._value2member_map_.values())
        return [_enum.value for _enum in _dict]


class HF_CONF(_CustomEnum):
    """Keys accepted by hftransformers hf_conf."""

    # TODO: check if this exists in evaluate-mlflow package
    CUSTOM_CONFIG_MODULE = "custom_config_module"
    CUSTOM_MODLE_MODULE = "custom_model_module"
    CUSTOM_TOKENIZER_MODULE = "custom_tokenizer_module"
    EXTRA_PIP_REQUIREMENTS = "extra_pip_requirements"
    FORCE_LOAD_CONFIG = "force_load_config"
    FORCE_LOAD_TOKENIZER = "force_load_tokenizer"
    HUGGINGFACE_ID = "huggingface_id"
    HF_CONFIG_ARGS = "config_hf_load_kwargs"
    HF_CONFIG_CLASS = "hf_config_class"
    HF_MODEL_ARGS = "model_hf_load_args"
    HF_PRETRAINED_CLASS = "hf_pretrained_class"
    HF_TOKENIZER_ARGS = "tokenizer_hf_load_kwargs"
    HF_TOKENIZER_CLASS = "hf_tokenizer_class"
    HF_PIPELINE_ARGS = "pipeline_init_args"
    HF_PREDICT_MODULE = "hf_predict_module"
    TASK_TYPE = "task_type"
    TRAIN_LABEL_LIST = "train_label_list"
    HF_MODEL_PATH = "model"
    HF_CONFIG_PATH = "config"
    HF_TOKENIZER_PATH = "tokenizer"
    HF_USE_EXPERIMENTAL_FEATURES = "exp"


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
    TEXT2TEXT_GENERATION = "text2text-generation"


class SupportedASRModelFamily(_CustomEnum):
    """Supported text to image models."""

    WHISPER = "whisper"


class SupportedTasks(_CustomEnum):
    """Supported Hugging face tasks for conversion to MLflow."""

    # NLP tasks
    FILL_MASK = "fill-mask"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TEXT_GENERATION = "text-generation"
    TEXT_CLASSIFICATION = "text-classification"
    TRANSLATION = "translation"
    # Vision tasks
    IMAGE_CLASSIFICATION = "image-classification"
    # ASR
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
