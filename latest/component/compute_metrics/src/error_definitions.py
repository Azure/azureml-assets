# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Error Definitions."""

from azureml._common._error_definition import error_decorator
from azureml._common._error_definition.user_error import (
    BadArgument,
    BadData,
    UserError
)
from azureml._common._error_definition.system_error import ClientError
from constants import ErrorStrings


class ModelEvaluationInternalError(ClientError):
    """Model Evaluation Internal Error.

    Args:
        ClientError (_type_): _description_
    """

    @property
    def message_format(self) -> str:
        """Message format."""
        return ErrorStrings.GenericModelEvaluationError


class ModelPredictionInternalError(ClientError):
    """Model Prediction error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericModelPredictionError


class ModelPredictionUserError(UserError):
    """Model Prediction error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericModelPredictionError


class ComputeMetricsInternalError(ClientError):
    """Compute Metrics error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericComputeMetricsError


class ComputeMetricsUserError(UserError):
    """Compute Metrics error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.GenericComputeMetricsError


@error_decorator(use_parent_error_code=True)
class DownloadDependenciesError(ClientError):
    """Download Dependencies error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.DownloadDependenciesFailed


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
class InvalidModel(BadArgument):
    """Model Validation error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidModel


@error_decorator(use_parent_error_code=True)
class BadModel(ClientError):
    """Invalid Model Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadModelData


@error_decorator(use_parent_error_code=True)
class InvalidData(BadArgument):
    """Invalid Test Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidData


@error_decorator(use_parent_error_code=True)
class InvalidFileInputSource(BadArgument):
    """Invalid Test Data Source error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidFileInputSource


@error_decorator(use_parent_error_code=True)
class InvalidPredictionColumnNameData(BadArgument):
    """Invalid Prediction Column Name data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidPredictionColumnNameData


@error_decorator(use_parent_error_code=True)
class ArgumentParsingError(BadArgument):
    """Argument Parsing Error error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.ArgumentParsingError


@error_decorator(use_parent_error_code=True)
class InvalidGroundTruthColumnNameData(BadArgument):
    """Invalid Ground Truth Column Name data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidGroundTruthColumnNameData


@error_decorator(use_parent_error_code=True)
class InvalidYTestCasesColumnNameData(BadArgument):
    """Invalid Ground Truth Column Name data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidYTestCasesColumnNameData


@error_decorator(use_parent_error_code=True)
class InvalidGroundTruthColumnNameCodeGen(BadArgument):
    """Invalid Ground Truth Column Name data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidGroundTruthColumnNameCodeGen


@error_decorator(use_parent_error_code=True)
class InvalidGroundTruthColumnName(BadArgument):
    """Ground Truth Column Name should be passed."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.InvalidGroundTruthColumnName


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


@error_decorator(use_parent_error_code=True)
class EmptyInputData(BadData):
    """Bad Input Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.EmptyInputData


@error_decorator(use_parent_error_code=True)
class BadInputColumnData(BadData):
    """Bad Input Column Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadInputColumnData


@error_decorator(use_parent_error_code=True)
class BadLabelColumnData(BadData):
    """Bad Label Column Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadLabelColumnName


@error_decorator(use_parent_error_code=True)
class BadQuestionsContextGroundTruthData(BadData):
    """Bad Questions and Context in Ground Truth Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadQuestionsContextGroundTruthData


@error_decorator(use_parent_error_code=True)
class BadFeatureColumnData(BadData):
    """Bad Feature Data error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadFeatureColumnNames


@error_decorator(use_parent_error_code=True)
class BadEvaluationConfigFile(BadData):
    """Bad Evaluation Config file."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadEvaluationConfigFile


@error_decorator(use_parent_error_code=True)
class BadEvaluationConfigParam(BadData):
    """Bad Evaluation Config param data."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadEvaluationConfigParam


@error_decorator(use_parent_error_code=True)
class BadEvaluationConfig(BadData):
    """Bad Evaluation Config data."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadEvaluationConfig


@error_decorator(use_parent_error_code=True)
class BadForecastData(BadInputData):
    """Bad Forecasting Data passed."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadForecastGroundTruthData


@error_decorator(use_parent_error_code=True)
class BadRegressionData(BadInputData):
    """Bad Regression Column type."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.BadRegressionColumnType


class MetricsLoggingError(ClientError):
    """Metrics Logging Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.MetricLoggingError


class FilteringDataError(ClientError):
    """Filtering Data Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.FilteringDataError


class SavingOutputError(ClientError):
    """Saving Output Failure error."""

    @property
    def message_format(self) -> str:
        """Message Format.

        Returns:
            str: _description_
        """
        return ErrorStrings.SavingOutputError
