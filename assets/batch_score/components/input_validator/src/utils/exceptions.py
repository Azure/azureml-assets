# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch Validation Exceptions"""

from typing import Optional


class AoaiBatchValidationErrorCode:
    """Aoai Batch Validation Error Codes"""

    INVALID_JSON_LINE = "invalid_json_line"
    MAXIMUM_REQUESTS_EXCEEDED = "maximum_requests_exceeded"
    MAXIMUM_INPUT_FILE_SIZE_EXCEEDED = "maximum_input_file_size_exceeded"
    INVALID_URL = "invalid_url"
    MISMATCHED_MODEL = "mismatched_model"
    MODEL_NOT_FOUND = "model_not_found"
    DUPLICATE_CUSTOM_ID = "duplicate_custom_id"
    TOKEN_LIMIT_EXCEEDED = "token_limit_exceeded"
    INVALID_REQUEST = "invalid_request"
    EMPTY_FILE = "empty_file"
    FILE_NOT_FOUND = "file_not_found"
    SERVER_ERROR = "server_error"
    # endpoint specific
    STREAMING_UNSUPPORTED = "streaming_unsupported"
    MAXIMUM_EMBEDDING_INPUTS_EXCEEDED = "maximum_embedding_inputs_exceeded"


class BatchValidationErrorMessage:
    """Batch Validation Error Messages"""

    INVALID_JSON_LINE = "The input line is not parsable as valid JSON."
    INVALID_URL = "The URL must be the same for all requests."
    MISMATCHED_MODEL = "The model must be the same for all requests."
    DUPLICATE_CUSTOM_ID = "The Custom ID must be unique for each request."
    INVALID_REQUEST = "The schema of the request is invalid."
    EMPTY_FILE = "The input file is empty."
    SERVER_ERROR = "The server encountered an error while processing the request."


class BatchValidationError:
    """Batch Validation Error"""

    code: AoaiBatchValidationErrorCode
    message: BatchValidationErrorMessage
    line: Optional[int] = None
    param: Optional[str] = None

    def __init__(
            self,
            code: AoaiBatchValidationErrorCode,
            message: BatchValidationErrorMessage,
            line: Optional[int] = None,
            param: Optional[str] = None
        ):
        self.code = code
        self.message = message
        self.line = line
        self.param = param
