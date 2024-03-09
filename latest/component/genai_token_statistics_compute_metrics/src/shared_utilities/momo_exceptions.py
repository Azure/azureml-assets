# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file define exceptions used in Model Monitoring pipeline."""


class DataNotFoundError(Exception):
    """Exception raised when data is not found."""

    def __init__(self, message):
        """Initialize a DataNotFoundError."""
        super().__init__(message)


class InvalidInputError(ValueError):
    """Exception raised when input is invalid."""

    def __init__(self, message):
        """Initialize a InvalidInputError."""
        super().__init__(message)
