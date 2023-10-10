# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import auto
from strenum import StrEnum
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)

# for consistency we use the same types as https://github.com/openai/evals/blob/main/evals/prompt/base.py
# these are designed to mirror the openai.Completion.create and openai.ChatCompletion.create APIs
OpenAICreatePrompt = Union[str, List[str], List[int], List[List[int]]]
# A message is a dictionary with "role" and "content" keys
OpenAIChatMessage = Dict[str, str]
# A chat log is a list of messages
OpenAICreateChatPrompt = List[OpenAIChatMessage]
OpenAICreate = Union[OpenAICreatePrompt, OpenAICreateChatPrompt]


class PromptType(StrEnum):
    completions = auto()
    chat = auto()


class Role(StrEnum):
    system = auto()
    user = auto()
    assistant = auto()


@dataclass
class Prompt(ABC):
    """
    Class for handling the creation of prompts for OpenAI's APIs.
    """

    @abstractmethod
    def to_openai_create_prompt(self):
        """
        Return the actual data to be passed as the `prompt` field to either `openai.ChatCompletion.create`,
        if the model is a chat model, or `openai.Completion.create` otherwise.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of words in the prompt."""
        pass


@dataclass
class CompletionsPrompt(Prompt):

    raw_prompt: OpenAICreatePrompt
    prompt_prefix = 'prompt'

    def __len__(self):
        return len(self.raw_prompt.split())

    def to_openai_create_prompt(self):
        return {"prompt": self.raw_prompt}


@dataclass
class ChatPrompt(Prompt):

    raw_prompt: OpenAICreateChatPrompt
    prompt_prefix = 'messages'

    def __len__(self):
        return sum(len(msg["content"].split()) for msg in self.raw_prompt)

    def to_openai_create_prompt(self):
        return {"prompt": self.raw_prompt}


