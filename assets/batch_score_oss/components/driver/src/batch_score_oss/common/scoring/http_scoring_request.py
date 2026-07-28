# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data class for HTTP scoring request."""

from dataclasses import dataclass, field


@dataclass
class HttpScoringRequest:
    """Http scoring request."""

    headers: dict = field(default_factory=dict)
    payload: str = None
    url: str = None
