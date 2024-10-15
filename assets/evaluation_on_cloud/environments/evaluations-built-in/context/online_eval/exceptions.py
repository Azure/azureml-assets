# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""File to create AzureML Based Exceptions for AOAI Evaluation."""

from azureml.exceptions import AzureMLException
from constants import ExceptionLiterals


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


class DataSavingException(ModelEvaluationException):
    """Data Saving Exception.

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
        target = ExceptionLiterals.DATA_SAVING_TARGET
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)


class AuthenticationException(ModelEvaluationException):
    """Authentication Exception."""

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
        target = ExceptionLiterals.AuthenticationException
        super().__init__(exception_message,
                         inner_exception=inner_exception,
                         target=target,
                         details=details,
                         message_format=message_format,
                         message_parameters=message_parameters,
                         reference_code=reference_code,
                         **kwargs)
