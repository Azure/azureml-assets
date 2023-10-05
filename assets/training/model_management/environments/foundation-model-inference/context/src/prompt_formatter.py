# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from typing import Dict
from typing import Union

from constants import TaskType
from conversation import Conversation


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
