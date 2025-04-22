# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exceptions util."""

import time
import logging
from functools import wraps
from azureml._common.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from azureml._common._error_definition.system_error import ClientError  # type: ignore
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


class NonMsiAttachedComputeError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.NON_MSI_ATTACHED_COMPUTE_ERROR


class UnSupportedModelTypeError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.UNSUPPORTED_MODEL_TYPE_ERROR


class MissingModelNameError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.MISSING_MODEL_NAME_ERROR


class ComputeCreationError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.COMPUTE_CREATION_ERROR


class EndpointCreationError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.ENDPOINT_CREATION_ERROR


class DeploymentCreationError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.DEPLOYMENT_CREATION_ERROR


class OnlineEndpointInvocationError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.ONLINE_ENDPOINT_INVOCATION_ERROR


class BatchEndpointInvocationError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.BATCH_ENDPOINT_INVOCATION_ERROR


class UserIdentityMissingError(ClientError):
    """Internal Import Model Generic Error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.USER_IDENTITY_MISSING_ERROR


class CondaEnvCreationError(ClientError):
    """Conda env creation error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.CONDA_ENV_CREATION_ERROR


class CondaFileMissingError(ClientError):
    """Conda file missing error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.CONDA_FILE_MISSING_ERROR


class MlflowModelValidationError(ClientError):
    """MLflow local validation error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.MLFLOW_LOCAL_VALIDATION_ERROR


class InvalidModelIDError(ClientError):
    """MLflow local validation error."""

    @property
    def message_format(self) -> str:
        """Message format."""
        return ModelImportErrorStrings.INVALID_MODEL_ID_ERROR


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
