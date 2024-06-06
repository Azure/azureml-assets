# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Chat Completion Request Body."""

from dataclasses import dataclass
from typing import List, Union

from batch_api.request_body.request_body import RequestBody
from batch_api.request_body.chat_completion.message import (
    SystemMessage,
    UserMessage
)


@dataclass
class ChatCompletionRequestBody(RequestBody):
    """Request body for chat completion."""

    messages: List[Union[SystemMessage, UserMessage]]
