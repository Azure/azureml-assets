# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exceptions util."""

import time
import logging
from functools import wraps
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException


class ModelImportErrorStrings:
    """Error strings."""

    LOG_SAFE_GENERIC_ERROR = "{pii_safe_message:log_safe}"
    LOG_UNSAFE_GENERIC_ERROR = "An error occurred: [{error}]"
    VALIDATION_ERROR = "Error while validating parameters [{error:log_safe}]"
    INVALID_HUGGING_FACE_MODEL_ID = (
        "Invalid Hugging face model id: {model_id}."
        " Please ensure that you are using a correct and existing model ID."
    )
    ERROR_FETCHING_HUGGING_FACE_MODEL_INFO = "Error in fetching model info for {model_id}. Error [{error}]"
    BLOBSTORAGE_DOWNLOAD_ERROR = "Failed to download artifacts from {uri}. Error: [{error}]"
    GIT_CLONE_ERROR = "Failed to clone {uri}. Error: [{error}]"
    GIT_CONFIG_ERROR = "Failed to set up GitHub config. Error: [{error}]"
    VM_NOT_SUFFICIENT_FOR_OPERATION = "VM not sufficient for {operation} operation. Details: [{details}]"
    CMD_EXECUTION_ERROR = "Error in executing command. Error: [{error}]"
    MODEL_ALREADY_EXISTS = "Model with name {model_id} already exists in registry {registry} at {url}"
    UNSUPPORTED_TASK_TYPE = "Unsupported task type {task_type} provided. Supported task types are {supported_tasks}."
    NON_MSI_ATTACHED_COMPUTE_ERROR = (
        "Kindly make sure that compute used by model_registration component"
        " has MSI(Managed Service Identity) associated with it."
        " Click here to know more -"
        " https://learn.microsoft.com/en-us/azure/machine-learning/"
        " how-to-identity-based-service-authentication?view=azureml-api-2&tabs=cli. Exception : {exception}"
    )
    USER_IDENTITY_MISSING_ERROR = (
        "Failed to get AzureMLOnBehalfOfCredential."
        " Kindly set UserIdentity as identity type if submitting job using sdk or cli."
        " Please take reference from given links :\n"
        " About - https://learn.microsoft.com/en-us/samples/azure/azureml-examples/azureml---on-behalf-of-feature/ \n"
        " sdk - https://aka.ms/azureml-import-model \n"
        " cli - https://aka.ms/obo-cli-sample"
    )
    HF_AUTHENTICATION_ERROR = (
        "Failed to authenticate with the HF Token provided: [{error}]"
    )


def swallow_all_exceptions(logger: logging.Logger):
    """Swallow all exceptions.

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
                if isinstance(e, MlException):
                    azureml_exception = e
                else:
                    message = ModelImportErrorStrings.LOG_UNSAFE_GENERIC_ERROR
                    azureml_exception = MlException(
                        message=message.format(error=e), no_personal_data_message=message,
                        error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.COMPONENT,
                        error=e
                    )

                logger.error("Exception {} when calling {}".format(azureml_exception, func.__name__))
                for handler in logger.handlers:
                    handler.flush()

                raise azureml_exception
            finally:
                time.sleep(60)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap
