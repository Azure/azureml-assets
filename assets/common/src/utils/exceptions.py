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
    NON_MSI_ATTACHED_COMPUTE_ERROR = (
        "Kindly make sure that compute used by model_registration component"
        " has MSI(Managed Service Identity) associated with it."
        " Click here to know more -"
        " https://learn.microsoft.com/en-us/azure/machine-learning/"
        " how-to-identity-based-service-authentication?view=azureml-api-2&tabs=cli. Exception : {exception}"
    )
    UNSUPPORTED_MODEL_TYPE_ERROR = "Unsupported model type : {model_type}"
    MISSING_MODEL_NAME_ERROR = "Missing Model Name. Provide model_name as input or in the model_download_metadata JSON"
    COMPUTE_CREATION_ERROR = "Error occured while creating compute cluster - {exception}"
    ENDPOINT_CREATION_ERROR = "Error occured while creating endpoint - {exception}"
    DEPLOYMENT_CREATION_ERROR = "Error occured while creating deployment - {exception}"
    ONLINE_ENDPOINT_INVOCATION_ERROR = "Invocation failed with error: {exception}"
    BATCH_ENDPOINT_INVOCATION_ERROR = "Invocation failed with error: {exception}"
    USER_IDENTITY_MISSING_ERROR = (
        "Failed to get AzureMLOnBehalfOfCredential."
        " Kindly set UserIdentity as identity type if submitting job using sdk or cli."
        " Please take reference from given links :\n"
        " About - https://learn.microsoft.com/en-us/samples/azure/azureml-examples/azureml---on-behalf-of-feature/ \n"
        " sdk - https://aka.ms/azureml-import-model \n"
        " cli - https://aka.ms/obo-cli-sample"
    )

    # local validation error strings
    CONDA_ENV_CREATION_ERROR = "Caught error while creating conda env using MLflow model conda.yaml"
    CONDA_FILE_MISSING_ERROR = (
        "Invalid MLflow model structure. Please make sure conda.yaml exists in MLflow model parent dir."
    )
    MLFLOW_LOCAL_VALIDATION_ERROR = (
        "Error in validating MLflow model. For more details please take a look at the previous logs."
    )
    INVALID_MODEL_ID_ERROR = (
        "Given Model ID : {model_id} is invalid. \n"
        "Model ID should follow one of this format :\n"
        "Workspace Model: \n"
        "    1) azureml:<model-name>:<version> \n"
        "    2) azureml://locations/<location>/workspaces/<workspace-name>/models/<model-name>/versions/<version> \n"
        "    3) /subscriptions/<subscription_id>/resourceGroups/<resource_group>/providers/"
        "Microsoft.MachineLearningServices/workspaces/<workspace-name>/models/<model-name>/versions/<version> \n"
        "Registry Model: \n"
        "    1) azureml://registries/<registry-name>/models/<model-name>/versions/<version> \n"
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
