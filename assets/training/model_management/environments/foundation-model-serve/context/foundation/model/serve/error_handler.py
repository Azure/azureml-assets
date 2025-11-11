# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Error handler module for Azure-compliant error responses.

This module provides utilities for creating standardized Azure error responses
that comply with Azure API error formatting standards.
"""
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from foundation.model.serve.api_server_setup.protocol import (
    AzureError,
)

CONTENT_LENGTH = "content-length"

status_code_mapping = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    409: "Conflict",
    422: "Invalid input",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


def get_status_code_string(status_code):
    """
    Return the code string corresponding to the given status code.

    Args:
        status_code (int): The HTTP status code to lookup.

    Returns:
        str: The corresponding status code description string.
    """
    return status_code_mapping.get(status_code, "Unknown Status Code")


def to_azure_error_json_response(status_code, message, headers):
    """Create an Azure standard error JSON response.

    Args:
        status_code (int): The HTTP status code.
        message: The error message.
        headers (dict): HTTP headers to include in the response.

    Returns:
        JSONResponse: A FastAPI JSONResponse with Azure-formatted error.
    """
    if headers is not None and CONTENT_LENGTH in headers:
        del headers[CONTENT_LENGTH]

    return JSONResponse(
        status_code=status_code,
        content={
            "error": jsonable_encoder(
                AzureError(
                    status=status_code,
                    code=get_status_code_string(status_code),
                    message=message,
                )
            )
        },
        headers=headers,
    )
