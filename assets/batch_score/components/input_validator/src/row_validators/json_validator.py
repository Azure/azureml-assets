# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Json Validator."""

import json

from row_validators.base_validator import BaseValidator
from row_validators.row_validation_context import RowValidationContext
from row_validators.row_validation_result import RowValidationResult
from utils.exceptions import (
    AoaiBatchValidationErrorCode,
    BatchValidationErrorMessage,
    BatchValidationError
)


class JsonValidator(BaseValidator):
    """Validate that the row is parsable as a valid JSON object."""

    def __init__(self) -> None:
        """Initialize the JsonValidator."""
        pass

    def validate_row(self, row_context: RowValidationContext) -> RowValidationResult:
        """Validate that the row is parsable as a valid JSON object."""
        result: RowValidationResult = RowValidationResult()

        try:
            json.loads(row_context.raw_input_row)

        except json.JSONDecodeError:
            result.error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.INVALID_JSON,
                message=BatchValidationErrorMessage.INVALID_JSON,
                line=row_context.line_number
            )

        return result
