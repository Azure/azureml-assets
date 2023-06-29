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


class ModelImportErrorStrings:
    """Error strings."""

    LOG_SAFE_GENERIC_ERROR = "{pii_safe_message:log_safe}"
    LOG_UNSAFE_GENERIC_ERROR = "An error occurred: [{error}]"
    VALIDATION_ERROR = "Error while validating parameters [{error:log_safe}]"           " "
    INVALID_HUGGING_FACE_MODEL_ID = (
        "Invalid Hugging face model id: {model_id}."
        " Please ensure that you are using a correct and existing model ID."
    )
    ERROR_FETCHING_HUGGING_FACE_MODEL_INFO = (
        "Error in fetching model info for {model_id}. Error [{error}]"
    )
    BLOBSTORAGE_DOWNLOAD_ERROR = "Failed to download artifacts from {uri}. Error: [{error}]"
    GIT_CLONE_ERROR = "Failed to clone {uri}. Error: [{error}]"


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

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.LOG_UNSAFE_GENERIC_ERROR


class InvalidHuggingfaceModelIDError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.INVALID_HUGGING_FACE_MODEL_ID


class GITCloneError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.GIT_CLONE_ERROR


class BlobStorageDownloadError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.BLOBSTORAGE_DOWNLOAD_ERROR


class InvalidHuggingfaceModelIDError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.INVALID_HUGGING_FACE_MODEL_ID


class HuggingFaceErrorInFetchingModelInfo(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.ERROR_FETCHING_HUGGING_FACE_MODEL_INFO



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
                if isinstance(e, AzureMLException):
                    azureml_exception = e
                else:
                    azureml_exception = AzureMLException._with_error(AzureMLError.create(ModelImportError, error=e))

                logger.error("Exception {} when calling {}".format(azureml_exception, func.__name__))
                for handler in logger.handlers:
                    handler.flush()
                raise azureml_exception
            finally:
                time.sleep(60)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap
