# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This module defines the constants and enums."""
from enum import Enum


class CommonConstants:
    """Common constants used across the module."""

    DEFAULT_PORT = 8000
    HOST = "localhost"
    CONTENT_SAFETY_THERESHOLD_DEFAULT = 2


class EnvironmentVariables:
    """Environment variables"""

    ENGINE_NAME = "ENGINE_NAME"
    AACS_INFERENCE_URI = "AACS_INFERENCE_URI"
    ENGINE_STARTUP_PORT = "ENGINE_STARTUP_PORT"
    ENGINE_STARTUP_FILE_PATH = "ENGINE_STARTUP_FILE_PATH"
    TASK_TYPE = "TASK_TYPE"
    AZUREML_MODEL_DIR = "AZUREML_MODEL_DIR"
    AML_MODEL = "AML_MODEL"
    AML_PORT = "AML_PORT"
    CONTENT_SAFETY_THRESHOLD = "CONTENT_SAFETY_THRESHOLD"
    UAI_CLIENT_ID = "UAI_CLIENT_ID"
    SUBSCRIPTION_ID = "SUBSCRIPTION_ID"
    RESOURCE_GROUP_NAME = "RESOURCE_GROUP_NAME"
    CONTENT_SAFETY_ACCOUNT_NAME = "CONTENT_SAFETY_ACCOUNT_NAME"
    CONTENT_SAFETY_ENDPOINT = "CONTENT_SAFETY_ENDPOINT"


class EngineName:
    VLLM = "VLLM"


class ExtraParameters:
    """Extra parameter options."""

    KEY = "extra-parameters"
    ERROR = "error"
    DROP = "drop"
    PASS_THROUGH = "pass-through"
    OPTIONS = [ERROR, DROP, PASS_THROUGH]


class OpenAIEndpoints:
    V1_COMPLETIONS = "v1/completions"
    V1_CHAT_COMLETIONS = "v1/chat/completions"


class TaskType(str, Enum):
    """Enum representing the types of tasks."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"

    def __str__(self):
        """Return the string representation of the task type."""
        return self.value
