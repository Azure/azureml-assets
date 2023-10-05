# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Dict
from typing import List, Union, Optional
from enum import Enum
from conversation import Conversation
from constants import TaskType


class AbstractPromptFormatter(ABC):
    @abstractmethod
    def format_prompt(
        self, task_type: TaskType, query: Union[str, Conversation], params: Dict
    ) -> str:
        pass


class Llama2Formatter(AbstractPromptFormatter):
    def format_prompt(
        self, task_type: TaskType, query: Union[str, Conversation], params: Dict
    ) -> str:
        # Formatting logic for Llama2 models, currently a placeholder.
        return str(query)
