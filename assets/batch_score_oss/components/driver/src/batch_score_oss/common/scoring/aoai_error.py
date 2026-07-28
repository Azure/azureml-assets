# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data class for Azure OpenAI scoring error."""

from dataclasses import dataclass


@dataclass
class AoaiScoringError:
    """Azure OpenAI scoring error."""

    code: str = None
    message: str = None
