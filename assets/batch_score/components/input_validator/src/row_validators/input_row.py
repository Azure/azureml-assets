# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Input Row"""

from dataclasses import dataclass
from typing import Union

from batch_api import (
    RequestBody,
    ChatCompletionRequestBody
)


@dataclass
class InputRow:
    """Expected Schema for an Individual Input Row"""

    custom_id: str
    method: str
    url: str
    body: Union[RequestBody, ChatCompletionRequestBody]
