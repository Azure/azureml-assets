# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Benchmarking related constants."""

from enum import Enum


class TaskType(Enum):
    CHAT_COMPLETION = "chat_completion"
    TEXT_GENERATION = "text_generation"
