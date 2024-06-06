# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Row Validation Result"""

from dataclasses import dataclass

from utils.exceptions import BatchValidationError


@dataclass
class RowValidationResult:
    """Row Validation Result"""

    error: BatchValidationError = None
    is_success: bool = error is None

    def __init__(self) -> None:
        self.error = None
