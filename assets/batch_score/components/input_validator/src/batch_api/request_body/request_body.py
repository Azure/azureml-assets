# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Request Body."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestBody:
    """Request Body."""

    model: str
    user: Optional[str]
