# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File to create AzureML Based Exceptions for Model Evaluation."""

from azureml.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from error_definitions import ModelEvaluationInternalError
from constants import ExceptionLiterals
from functools import wraps
import time
import logging


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
                if isinstance(e, AzureMLException):
                    azureml_exception = e
                else:
                    azureml_exception = AzureMLException._with_error(
                        AzureMLError.create(ModelEvaluationInternalError, error=e))

                logger.error("Exception {} when calling {}".format(azureml_exception, func.__name__))
                for handler in logger.handlers:
                    handler.flush()
                raise azureml_exception
            finally:
                time.sleep(60)  # Let telemetry logger flush its logs before terminating.

        return wrapper

    return wrap


class ModelEvaluationException(AzureMLException):
    """Base Model Evaluation Exception."""

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 target=None, details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            target (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        if not target:
            target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class ArgumentValidationException(ModelEvaluationException):
    """Argument Validation Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.ARGS_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class DataValidationException(ModelEvaluationException):
    """Data Validation Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.DATA_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class DataLoaderException(ModelEvaluationException):
    """Data Loader Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.DATA_LOADING_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class ModelValidationException(ModelEvaluationException):
    """Model Validation Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.MODEL_LOADER_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class ScoringException(ModelEvaluationException):
    """Score Mode Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class PredictException(ModelEvaluationException):
    """Predict Mode Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class ComputeMetricsException(ModelEvaluationException):
    """Compute Metrics Mode Exception.

    Args:
        ModelEvaluationException (_type_): _description_
    """

    def __init__(self,
                 exception_message,
                 inner_exception=None,
                 details=None,
                 message_format=None,
                 message_parameters=None,
                 reference_code=None,
                 **kwargs):
        """__init__.

        Args:
            exception_message (_type_): _description_
            inner_exception (_type_, optional): _description_. Defaults to None.
            details (_type_, optional): _description_. Defaults to None.
            message_format (_type_, optional): _description_. Defaults to None.
            message_parameters (_type_, optional): _description_. Defaults to None.
            reference_code (_type_, optional): _description_. Defaults to None.
        """
        target = ExceptionLiterals.MODEL_EVALUATION_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)
