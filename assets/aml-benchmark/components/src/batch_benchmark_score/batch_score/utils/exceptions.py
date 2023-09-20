# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exceptions util."""

import time
import logging
import traceback
from functools import wraps
from typing import Callable, Any

from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml._common._error_response._error_response_constants import ErrorCodes
from azureml.core.run import Run

from .error_definitions import BenchmarkSystemError


class BenchmarkException(AzureMLException):
    """Base exception for Benchmark."""


# System Exception
class BenchmarkSystemException(BenchmarkException):
    """Exception for internal errors that happen within the SDK."""

    _error_code = ErrorCodes.SYSTEM_ERROR


# User Exception
class BenchmarkUserException(BenchmarkException):
    """Exception for user errors caught during Benchmarking."""

    _error_code = ErrorCodes.USER_ERROR


class BenchmarkValidationException(BenchmarkUserException):
    """Exception for any errors caught when validating inputs."""

    _error_code = ErrorCodes.VALIDATION_ERROR
