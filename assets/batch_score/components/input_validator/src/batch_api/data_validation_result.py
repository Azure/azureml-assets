# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Input Row."""

from dataclasses import dataclass
from typing import List

from batch_api.batch_reference import BatchReference
from utils.exceptions import BatchValidationError


@dataclass
class DataValidationResult:
    """Result of the Data Validation to send to MBI."""

    batch_reference: BatchReference
    deployment_name: str
    endpoint_url: str
    input_token_count: float
    request_count: int
    errors: List[BatchValidationError]

    def __init__(self):
        """Initialize the DataValidationResult object."""
        self.errors = []
