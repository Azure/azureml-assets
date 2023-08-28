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


class DatasetDownloadException(BenchmarkUserException):
    """Exception for any errors caught when downloading datasets."""

    _error_code = "DatasetDownloadError"


class DataFormatException(BenchmarkUserException):
    """Exception for any errors related to data format."""

    _error_code = ErrorCodes.DATAFORMAT_ERROR


class MissingColumnException(BenchmarkUserException):
    """Exception for any errors related to missing columns."""

    _error_code = ErrorCodes.MISSINGCOLUMN_ERROR


def swallow_all_exceptions(logger: logging.Logger) -> Callable[..., Any]:
    """
    Swallow all exceptions.

    1. Catch all the exceptions arising in the functions wherever used
    2. Raise the exception as an AzureML Exception so that it does not get scrubbed by PII scrubber

    :param logger: The logger to be used for logging the exception raised
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, AzureMLException):
                    azureml_exception = e
                else:
                    azureml_exception = BenchmarkSystemException._with_error(
                        AzureMLError.create(
                            BenchmarkSystemError,
                            error_details=str(e),
                            traceback=traceback.format_exc(),
                        ),
                        inner_exception=e,
                    )
                logger.error(azureml_exception.message)
                for handler in logger.handlers:
                    handler.flush()
                run = Run.get_context()
                logger.info("Marking run as failed...")
                run.fail(error_details=azureml_exception)
                raise
            finally:
                time.sleep(30)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap
