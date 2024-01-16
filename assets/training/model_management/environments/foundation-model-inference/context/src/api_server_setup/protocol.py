# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import time
from vllm.entrypoints.openai.protocol import ChatMessage, UsageInfo

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from vllm.utils import random_uuid


class ChatCompletionResponseChoice(BaseModel):
    """Class for one of the 'choices' in the openai api chat completion response."""

    index: int = 0
    message: Optional[ChatMessage] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class ChatCompletionResponse(BaseModel):
    """An Openai chat completion response object."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo


class LogProbs(BaseModel):
    """Class that represents logprobs in the openai way."""

    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    """Class for one of the 'choices' in the openai api text generation response."""

    index: int = 0
    text: str = ""
    logprobs: Optional[LogProbs] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class CompletionResponse(BaseModel):
    """An openai text generation response object."""

    id: str = Field(default_factory=lambda: f"cmpl-{random_uuid()}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
