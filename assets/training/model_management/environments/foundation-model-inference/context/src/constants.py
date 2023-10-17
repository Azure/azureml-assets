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

    def __str__(self):
        """Return the string representation of the task type."""
        return self.value


class SupportedTask:
    """Supported tasks by text-generation-inference."""

    TEXT_GENERATION = "text-generation"
    CHAT_COMPLETION = "chat-completion"


class ServerSetupParams:
    """Parameters for setting up the server."""

    WAIT_TIME_MIN = 15  # time to wait for the server to become healthy
