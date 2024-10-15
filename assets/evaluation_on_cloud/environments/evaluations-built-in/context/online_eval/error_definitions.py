# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Error Definitions."""

from azureml._common._error_definition import error_decorator
from azureml._common._error_definition.user_error import (
    BadData,
    UserError
)
from azureml._common._error_definition.system_error import ClientError
from constants import ErrorStrings


class OnlineEvalInternalError(ClientError):
    """Online Eval Internal Error.

    Args:
        ClientError (_type_): _description_
    """

    @property
    def message_format(self) -> str:
        """Message format."""
        return ErrorStrings.GenericOnlineEvalError


class OnlineEvalUserError(UserError):
    """OnlineEval error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericOnlineEvalError


@error_decorator(use_parent_error_code=True)
class BadInputData(BadData):
    """Bad Input Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadInputData


class SavingOutputError(ClientError):
    """Saving Output Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.SavingOutputError


class NoDataFoundError(ClientError):
    """Saving Output Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.DataNotFound


class OnlineEvalAuthError(UserError):
    """ONLINEAggregators error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.OnlineEvalAuthError


class OnlineEvalQueryError(UserError):
    """ONLINEAggregators error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.OnlineEvalQueryError
