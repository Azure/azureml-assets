# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Exceptions."""
from azureml.exceptions import AzureMLException
from constants import ExceptionLiterals


# Base Exception
class AutoPromptException(AzureMLException):
    """AutoPrompt Exception."""

    def __init__(self, exception_message, **kwargs):
        """__init__."""
        if kwargs.get("target", None):
            kwargs["target"] = ExceptionLiterals.AutoPromptTarget
        super().__init__(exception_message, **kwargs)


class ComputeMetricsException(AutoPromptException):
    """Compute Metrics Exception."""

    def __init__(self, exception_message, **kwargs):
        """__init__."""
        kwargs["target"] = ExceptionLiterals.MetricsPackageTarget
        super().__init__(exception_message, **kwargs)


class ArgumentValidationException(AutoPromptException):
    """Argument Validation Exception."""

    def __init__(self, exception_message, **kwargs):
        """__init__."""
        kwargs["target"] = ExceptionLiterals.ArgumentValidationTarget
        super().__init__(exception_message, **kwargs)


class DataLoaderException(AutoPromptException):
    """Data Loader Exception."""

    def __init__(self, exception_message, **kwargs):
        """__init__."""
        kwargs["target"] = ExceptionLiterals.DataLoaderTarget
        super().__init__(exception_message, **kwargs)
