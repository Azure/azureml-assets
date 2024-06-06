# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Message"""

from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    """Message"""

    role: str


@dataclass
class SystemMessage(Message):
    """SystemMessage"""

    content: str


@dataclass
class UserMessage(Message):
    """UserMessage"""

    content: Any