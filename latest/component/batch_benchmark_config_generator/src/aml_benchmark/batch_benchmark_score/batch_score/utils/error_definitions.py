# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Errors for Benchmarking."""

from azureml._common._error_definition.system_error import SystemError
from azureml._common._error_definition.user_error import UserError

from .error_strings import BenchmarkErrorStrings


class BenchmarkSystemError(SystemError):
    """Top level unknown system error."""

    @property
    def message_format(self) -> str:
        """Non-formatted error message."""
        return BenchmarkErrorStrings.INTERNAL_ERROR


class BenchmarkUserError(UserError):
    """Generic User Error."""

    @property
    def message_format(self) -> str:
        """Message Format."""
        return BenchmarkErrorStrings.GENERIC_ERROR


class BenchmarkValidationError(BenchmarkUserError):
    """Benchmark Validation User Error."""
