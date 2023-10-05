# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum


class EngineName(str, Enum):
    HF = "hf"
    VLLM = "vllm"
    MII = "mii"

    def __str__(self):
        return self.value


class TaskType(str, Enum):
    TEXT_GENERATION = "text-generation"
    CONVERSATIONAL = "conversational"

    def __str__(self):
        return self.value
