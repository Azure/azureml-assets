# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Schema Validator"""

import json
from typing import Union

from batch_api import (
    RequestBody, ChatCompletionRequestBody, SystemMessage, UserMessage
)
from row_validators.base_validator import BaseValidator
from row_validators.input_row import InputRow
from row_validators.row_validation_context import RowValidationContext
from row_validators.row_validation_result import RowValidationResult
from utils.exceptions import (
    AoaiBatchValidationErrorCode,
    BatchValidationErrorMessage,
    BatchValidationError
)


class SchemaValidator(BaseValidator):
    """Validates that the row is a JSON object with the expected schema"""

    def __init__(self) -> None:
        pass

    def validate_row(self, row_context: RowValidationContext) -> RowValidationResult:
        """Validates that the row is a JSON object with the expected schema"""

        result: RowValidationResult = RowValidationResult()

        try:
            input_row_dict = json.loads(row_context.raw_input_row)
            row_context.parsed_input_row = self.get_valid_input_row(input_row_dict)

        except:
            result.error = BatchValidationError(
                code=AoaiBatchValidationErrorCode.INVALID_REQUEST,
                message=BatchValidationErrorMessage.INVALID_REQUEST,
                line=row_context.line_number
            )

        return result

    def get_valid_input_row(self, data: dict) -> InputRow:
        """Get a valid input row object from a dictionary"""

        return InputRow(
            custom_id=data["custom_id"],
            method=data["method"],
            url=data["url"],
            body=self.get_valid_request_body(data["body"])
        )

    def get_valid_request_body(self, data: dict) -> Union[RequestBody, ChatCompletionRequestBody]:
        """Get a valid request body object from a dictionary"""

        if "messages" in data:
            return ChatCompletionRequestBody(
                model=data["model"],
                user=data.get("user", None),
                messages=[self.get_valid_message(message) for message in data["messages"]]
            )

        else:
            return RequestBody(
                model=data["model"],
                user=data.get("user", None)
            )

    def get_valid_message(self, data: dict) -> Union[SystemMessage, UserMessage]:
        """Get a valid message object from a dictionary"""

        if data["role"] in ["system", "assistant", "tool"]:
            return SystemMessage(
                role=data["role"],
                content=data["content"]
            )

        elif data["role"] == "user":
            return UserMessage(
                role=data["role"],
                content=data["content"]
            )

        else:
            raise ValueError("Invalid message role")