# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Module for formatting prompts for different tasks and models."""
from abc import ABC, abstractmethod
from typing import Dict, Union

from constants import TaskType
from conversation import Conversation


class AbstractPromptFormatter(ABC):
    """Abstract base class for prompt formatters."""

    @abstractmethod
    def format_prompt(
        self,
        task_type: TaskType,
        query: Union[str, Conversation],
        params: Dict,
    ) -> str:
        """Format the prompt based on the task type, query and parameters."""
        raise NotImplementedError("format_prompt method not implemented.")


class Llama2Formatter(AbstractPromptFormatter):
    """Prompt formatter for Llama2 models."""

    def format_prompt(
        self,
        task_type: TaskType,
        query: Union[str, Conversation],
        params: Dict,
    ) -> str:
        """Format the prompt for Llama2 models. Currently a placeholder."""
        return str(query)
