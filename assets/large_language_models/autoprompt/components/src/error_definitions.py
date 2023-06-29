# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Error Definitions."""

from azureml._common._error_definition import error_decorator
from azureml._common._error_definition.user_error import (
    BadArgument,
    BadData
)
from azureml._common._error_definition.system_error import ClientError
from constants import ErrorStrings


@error_decorator(use_parent_error_code=True)
class AutoPromptInternalError(ClientError):
    """AutoPrompt Internal Error.

    Args:
        ClientError (_type_): _description_
    """

    @property
    def message_format(self) -> str:
        """Message format."""
        return ErrorStrings.GenericAutoPromptError


@error_decorator(use_parent_error_code=True)
class OpenAIModuleError(ClientError):
    """OpenAI Package Internal Error.

    Args:
        ClientError (_type_): _description_
    """

    @property
    def message_format(self) -> str:
        """Message format."""
        return ErrorStrings.GenericOpenAIError


@error_decorator(use_parent_error_code=True)
class OpenAIInitError(ClientError):
    """OpenAI Init Error.

    Args:
        ClientError (_type_): _description_
    """

    @property
    def message_format(self) -> str:
        """Message format."""
        self.is_transient = True
        return ErrorStrings.OpenAIInitError


@error_decorator(use_parent_error_code=True)
class ComputeMetricsInternalError(AutoPromptInternalError):
    """Compute Metrics error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericComputeMetricsError


@error_decorator(use_parent_error_code=True)
class InvalidTaskType(BadArgument):
    """Task Validation error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidTaskType


@error_decorator(use_parent_error_code=True)
class InvalidQuestionsKey(BadArgument):
    """Invalid Questions key error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidQuestionsKey


@error_decorator(use_parent_error_code=True)
class InvalidAnswersKey(BadArgument):
    """Invalid Answers key error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidAnswersKey


@error_decorator(use_parent_error_code=True)
class InvalidContextKey(BadArgument):
    """Invalid context key error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidContextKey


@error_decorator(use_parent_error_code=True)
class BadInputData(BadData):
    """Bad Test Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadTestData


@error_decorator(use_parent_error_code=True)
class MetricsLoggingError(ClientError):
    """Metrics Logging Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.MetricLoggingError
