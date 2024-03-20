# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exceptions util."""

import os
import sys
import time
import logging
import traceback
import inspect
from functools import wraps
from typing import Callable, Any

from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml._common._error_response._error_response_constants import ErrorCodes

from .error_definitions import BenchmarkSystemError
from .logging import custom_dimensions, run_details, log_traceback
from .constants import ROOT_RUN_PROPERTIES


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
            def get_calling_module_folder():
                """Get the folder name of the main script called from the command line."""
                main_script_path = inspect.stack()[-1].filename
                main_folder_name = os.path.basename(os.path.dirname(main_script_path))
                return main_folder_name

            try:
                custom_dimensions.app_name = get_calling_module_folder()
                root_run = run_details.root_run
                try:
                    root_run.add_properties(properties=ROOT_RUN_PROPERTIES)
                except Exception:
                    pass  # when already added by other child runs
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
                log_traceback(azureml_exception, logger, azureml_exception.message)             
                logger.info("Marking run as failed...")
                for handler in logger.handlers:
                    handler.flush()
                run_details.run.fail(error_details=azureml_exception)
                raise
            finally:
                time.sleep(30)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap
