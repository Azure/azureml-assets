# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common Property Validator."""

from row_validators.base_validator import BaseValidator
from row_validators.row_validation_context import RowValidationContext
from row_validators.row_validation_result import RowValidationResult
from utils.exceptions import (
    AoaiBatchValidationErrorCode,
    BatchValidationErrorMessage,
    BatchValidationError
)


class CommonPropertyValidator(BaseValidator):
    """Validate that the details of the request are consistent with those of the previous requests."""

    def __init__(self) -> None:
        """Initialize the CommonPropertyValidator."""
        self._custom_ids: set[str] = set()
        self._model: str = None
        self._url: str = None

    def validate_row(self, row_context: RowValidationContext) -> RowValidationResult:
        """Validate that the details of the request are consistent with those of the previous requests."""
        result: RowValidationResult = RowValidationResult()

        self.validate_custom_id(row_context, result)
        self.validate_model(row_context, result)
        self.validate_url(row_context, result)

        return result

    def validate_custom_id(self, row_context: RowValidationContext, result: RowValidationResult):
        """Validate that the request's Custom ID is unique."""
        if row_context.parsed_input_row.custom_id in self._custom_ids:
            result.error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.DUPLICATE_CUSTOM_ID,
                message=BatchValidationErrorMessage.DUPLICATE_CUSTOM_ID,
                line=row_context.line_number
            )

        self._custom_ids.add(row_context.parsed_input_row.custom_id)

    def validate_model(self, row_context: RowValidationContext, result: RowValidationResult):
        """Validate that the request has the same model as previous requests."""
        if self._model is None:
            self._model = row_context.parsed_input_row.body.model

        if row_context.parsed_input_row.body.model != self._model:
            result.error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.MISMATCHED_MODEL,
                message=BatchValidationErrorMessage.MISMATCHED_MODEL,
                line=row_context.line_number
            )

    def validate_url(self, row_context: RowValidationContext, result: RowValidationResult):
        """Validate that the request has the same URL as previous requests."""
        if self._url is None:
            self._url = row_context.parsed_input_row.url

        if row_context.parsed_input_row.url != self._url:
            result.error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.INVALID_URL,
                message=BatchValidationErrorMessage.INVALID_URL,
                line=row_context.line_number
            )
