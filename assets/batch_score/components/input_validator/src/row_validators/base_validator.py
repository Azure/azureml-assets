# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Base Row Validator"""

from abc import ABC, abstractmethod

from row_validators.row_validation_context import RowValidationContext
from row_validators.row_validation_result import RowValidationResult


class BaseValidator(ABC):
    """Base Row Validator"""

    @abstractmethod
    def validate_row(self, row_context: RowValidationContext) -> RowValidationResult:
        """Validate the row"""
        pass
