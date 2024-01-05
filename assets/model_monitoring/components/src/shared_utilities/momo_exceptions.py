# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file define exceptions used in Model Monitoring pipeline."""


class UserError(Exception):
    """Exception raised when it is user error."""

    def __init__(self, message):
        """Initialize a DataNotFoundError."""
        super().__init__(f'UserError: {message}')


class DataNotFoundError(UserError):
    """Exception raised when data is not found."""

    def __init__(self, message):
        """Initialize a DataNotFoundError."""
        super().__init__(message)


class InvalidInputError(ValueError):
    """Exception raised when input is invalid."""

    def __init__(self, message):
        """Initialize a InvalidInputError."""
        super().__init__(message)
