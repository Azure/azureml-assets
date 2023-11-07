# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module defines the EngineName and TaskType enums."""

from enum import Enum


class EngineName(str, Enum):
    """Enum representing the names of the engines."""

    HF = "hf"
    VLLM = "vllm"
    MII = "mii"

    def __str__(self):
        """Return the string representation of the engine name."""
        return self.value


class TaskType(str, Enum):
    """Enum representing the types of tasks."""

    TEXT_GENERATION = "text-generation"
    CONVERSATIONAL = "conversational"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION_CODE = "text-generation-code"
    FILL_MASK = "fill-mask"
    CHAT_COMPLETION = "chat-completion"

    def __str__(self):
        """Return the string representation of the task type."""
        return self.value


class SupportedTask:
    """Supported tasks by text-generation-inference."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"
    TEXT_CLASSIFICATION = "text-classification"
    TEXT_CLASSIFICATION_MULTILABEL = "text-classification-multilabel"
    NER = "text-named-entity-recognition"
    SUMMARIZATION = "text-summarization"
    QnA = "question-answering"
    TRANSLATION = "text-translation"
    TEXT_GENERATION_CODE = "text-generation-code"
    FILL_MASK = "fill-mask"

MLTABLE_FILE_NAME = "MLTable"
LLM_FT_PREPROCESS_FILENAME = "preprocess_args.json"
LLM_FT_TEST_DATA_KEY = "raw_test_data_fname"


class ServerSetupParams:
    """Parameters for setting up the server."""

    WAIT_TIME_MIN = 15  # time to wait for the server to become healthy


class VLLMSupportedModels:
    """VLLM Supported Models List."""

    Models = {
        "AquilaForCausalLM",
        "BaiChuanForCausalLM",
        "BloomForCausalLM",
        "FalconForCausalLM",
        "GPT2LMHeadModel",
        "GPTBigCodeForCausalLM",
        "GPTJForCausalLM",
        "GPTNeoXForCausalLM",
        "InternLMForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "MPTForCausalLM",
        "OPTForCausalLM",
        "QWenLMHeadModel"
    }


class MIISupportedModels:
    """MII Supported Models."""
    # TODO: Add more models from different tasks

    Models = {
        "BloomForCausalLM",
        "GPT2LMHeadModel",
        "GPTBigCodeForCausalLM",
        "GPTJForCausalLM",
        "GPTNeoXForCausalLM",
        "LlamaForCausalLM",
        "OPTForCausalLM",
    }


ALL_TASKS = [
    SupportedTask.TEXT_CLASSIFICATION,
    SupportedTask.TEXT_CLASSIFICATION_MULTILABEL,
    SupportedTask.NER,
    SupportedTask.SUMMARIZATION,
    SupportedTask.QnA,
    SupportedTask.TRANSLATION,
    SupportedTask.FILL_MASK,
    SupportedTask.TEXT_GENERATION,
    SupportedTask.CHAT_COMPLETION,
]

MULTILABEL_SET = [
    SupportedTask.TEXT_CLASSIFICATION_MULTILABEL,
]

CLASSIFICATION_SET = [
    SupportedTask.TEXT_CLASSIFICATION,
    SupportedTask.TEXT_CLASSIFICATION_MULTILABEL
]

MULTIPLE_OUTPUTS_SET = [
    SupportedTask.NER,
    SupportedTask.TEXT_CLASSIFICATION_MULTILABEL
]
