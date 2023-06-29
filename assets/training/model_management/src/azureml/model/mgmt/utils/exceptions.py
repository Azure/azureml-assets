# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exception Util."""

import time
import logging
from functools import wraps
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from azureml._common._error_definition import error_decorator  # type: ignore
from azureml._common._error_definition.system_error import ClientError  # type: ignore


class ModelImportException(AzureMLException):
    """Base exception for Model Import handling."""

    def __init__(self, exception_message, **kwargs):
        """Initialize a new instance of LLMException.

        :param exception_message: A message describing the error
        :type exception_message: str
        """
        super(ModelImportException, self).__init__(exception_message, **kwargs)

    @property
    def error_code(self):
        """Return error code for azureml_error."""
        return self._azureml_error.error_definition.code


class ModelImportError(ClientError):
    """Internal Import Model Generic Error."""

    GENERIC_ERROR = "An error occurred [{error}]"

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportError.GENERIC_ERROR


def swallow_all_exceptions(logger: logging.Logger):
    """
    Swallow all exceptions
    1. Catch all the exceptions arising in the functions wherever used
    2. Raise the exception as an AzureML Exception so that it does not get scrubbed by PII scrubber
    :param logger: The logger to be used for logging the exception raised
    :type logger: Instance of logging.logger
    """

    def wrap(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # This will be logged to exceptions table
                azureml_exception = AzureMLException._with_error(AzureMLError.create(ModelImportError, error=e))
                logger.error("Exception {} when calling {}".format(azureml_exception, func.__name__))
                for handler in logger.handlers:
                    handler.flush()
                raise azureml_exception
            finally:
                time.sleep(60)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap
