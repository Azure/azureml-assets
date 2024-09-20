# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data class for Azure OpenAI scoring response."""

from dataclasses import dataclass


@dataclass
class AoaiScoringResponse:
    """Azure OpenAI scoring response."""

    body: any = None
    request_id: str = None
    status_code: int = None
