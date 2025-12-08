# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for logging."""

from typing import Any, Dict, Union, Tuple
from importlib.metadata import entry_points
import logging
import os
import sys
import traceback
from collections import deque
import hashlib
import uuid
import platform
import json
import traceback
import sys

import mlflow
import azureml.core
from azureml.exceptions import AzureMLException
from azureml.core import Run
from azureml.telemetry.logging_handler import get_appinsights_log_handler
from azureml.telemetry import INSTRUMENTATION_KEY

from .constants import LoggerConfig, ExceptionTypes
from .aml_run import RunDetails


AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT = "aml_benchmark.azureml_benchmark_custom_logger"


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(
        self,
        run_details,
        app_name=LoggerConfig.DEFAULT_MODULE_NAME,
    ) -> None:
        """Init Custom dimensions."""
        self.app_name = app_name
        self.common_core_version = azureml.core.__version__

        # run_details
        self.run_id = run_details.run.id
        self.parent_run_id = run_details.parent_run.id
        self.root_run_id = run_details.root_run.id
        self.experiment_id = run_details.experiment.id
        self.subscription_id = run_details.subscription
        self.workspace_name = run_details.workspace.name
        self.root_attribution = run_details.root_attribute
        self.region = run_details.region
        self.compute_target = run_details.compute

        # component execution info
        self.os_info = platform.system()

        # additional info
        run_info: Dict[str, str] = run_details.get_extra_run_info()
        self.moduleName = run_info.get("moduleName", "")
        self.moduleId = run_info.get("moduleId", "")
        self.pipeline_type = run_info.get("pipeline_type", "")
        self.source = run_info.get("source", "")
        self.location = run_info.get("location", "")


class AppInsightsPIIStrippingFormatter(logging.Formatter):
    """Formatter for App Insights Logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format incoming log record.

        Args:
            record (logging.LogRecord): _description_

        Returns:
            str: _description_
        """
        exception_tb = getattr(record, 'exception_tb_obj', None)
        if exception_tb is None:
            return super().format(record)

        not_available_message = '[Not available]'
        properties = getattr(record, 'properties', {})

        message = properties.get('message', LoggerConfig.NON_PII_MESSAGE)
        # TODO: Update the logic later. Right now, prevent logging error message
        message = LoggerConfig.NON_PII_MESSAGE
        traceback_msg = properties.get('exception_traceback', not_available_message)

        record.message = record.msg = '\n'.join([
            'Type: {}'.format(properties.get('error_type', ExceptionTypes.Unclassified)),
            'Class: {}'.format(properties.get('exception_class', not_available_message)),
            'Message: {}'.format(message),
            'Traceback: {}'.format(traceback_msg),
            'ExceptionTarget: {}'.format(properties.get('exception_target', not_available_message))
        ])

        # Update exception message and traceback in extra properties as well
        properties['exception_message'] = message

        return super().format(record)


class AMLBenchmarkHandler(logging.StreamHandler):
    """aml benchmark handler for stream handling."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit logs to stream after adding custom dimensions."""
        new_properties = getattr(record, "properties", {})
        new_properties.update({'log_id': str(uuid.uuid4())})
        custom_dims_dict = custom_dimensions.__dict__
        cust_dim_copy = custom_dims_dict.copy()
        cust_dim_copy.update(new_properties)
        setattr(record, "properties", cust_dim_copy)
        msg = self.format(record)
        stream = self.stream
        stream.write(msg)


run_details = RunDetails()
custom_dimensions = CustomDimensions(run_details=run_details)


def log_mlflow_params(**kwargs: Any) -> None:
    """
    Log the provided key-value pairs as parameters in MLflow.

    If a file path or a list of file paths is provided in the value, the checksum of the
    file(s) is calculated and logged as the parameter value. If `None` is provided as
    the value, the parameter is not logged.

    :param **kwargs: Key-value pairs of parameters to be logged in MLflow.
    :return: None
    """
    MLFLOW_PARAM_VALUE_MAX_LEN = 500
    OVERFLOW_STR = '...'
    params = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and os.path.isfile(value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(open(value, "rb").read()).hexdigest()
            params[key] = checksum
        elif isinstance(value, list) and all(isinstance(item, str) and os.path.isfile(item) for item in value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(b"".join(open(item, "rb").read() for item in value)).hexdigest()
            params[key] = checksum
        else:
            if value is not None:
                if isinstance(value, str) and len(value) > MLFLOW_PARAM_VALUE_MAX_LEN:
                    value_len = MLFLOW_PARAM_VALUE_MAX_LEN - len(OVERFLOW_STR)
                    params[key] = value[: value_len] + OVERFLOW_STR
                else:
                    params[key] = value

    mlflow.log_params(params)

import traceback

class BufferStore:
    # create a maximum 10000 lines deque to store the logs
    _data : "deque[str]" = deque(maxlen=10000)

    @classmethod
    def push_data(cls, value: str):
        """Append a string to the list of strings stored under the given key."""
        cls._data.append(value)

    @classmethod
    def clear_buffer(cls):
        """Clear the buffer."""
        cls._data.clear()

    @classmethod
    def get_all_data(cls):
        """Return all the data stored."""
        return "\n".join(cls._data)
    

# implement a logging handler that appends to the BufferStore
class BufferStoreHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        BufferStore.push_data(msg)


def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger to always print logs on the stdout console.

    This function creates a logger with the specified filename and configures it
    by setting the logging level to INFO, adding a StreamHandler to the logger
    that outputs to stdout, and specifying a specific log message format.

    :param filename: The name of the file associated with the logger.
    :param level: Verbosity level for the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers to avoid duplicates

    formatter = logging.Formatter(
        "SystemLog: [%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )

    # Create a StreamHandler for stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    # create a BufferStoreHandler for storing logs
    buffer_store_handler = BufferStoreHandler()
    buffer_store_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(buffer_store_handler)

    return logger


def _get_error_details(
        exception: BaseException, logger: Union[logging.Logger, logging.LoggerAdapter]
) -> Tuple[str, str, str]:
    """
    Extract the error details from the base exception.

    For exceptions outside AzureML (e.g. Python errors), all properties are set as 'Unclassified'

    :param exception: The exception from which to extract the error details
    :param logger: The logger object to log to
    :return: An error code, error type (i.e. UserError or SystemError) and exception's target
    """
    default_target = "Unspecified"
    error_code = ExceptionTypes.Unclassified
    error_type = ExceptionTypes.Unclassified
    exception_target = default_target

    if isinstance(exception, AzureMLException):
        try:
            serialized_ex = json.loads(exception._serialize_json())
            error = serialized_ex.get(
                "error", {"code": ExceptionTypes.Unclassified, "inner_error": {}, "target": default_target}
            )

            # This would be the complete hierarchy of the error
            error_code = str(error.get("inner_error", ExceptionTypes.Unclassified))

            # This is one of 'UserError' or 'SystemError'
            error_type = error.get("code")

            exception_target = error.get("target")
            return error_code, error_type, exception_target
        except Exception:
            logger.warning(
                "Failed to parse error details while logging traceback from exception of type {}".format(exception)
            )

    return error_code, error_type, exception_target


def _log_traceback(exception: Union[AzureMLException, BaseException], logger, message=None):
    """Log exceptions without PII in APP Insights and full tracebacks in logger.

    Args:
        exception (_type_): _description_
        logger (_type_): _description_
        message (_type_): _description_
    """
    exception_message = "No message available."
    if hasattr(exception, "message"):
        exception_message = exception.message
    elif hasattr(exception, "exception_message"):
        exception_message = exception.exception_message
    message = exception_message if message is None else "\n".join([message, exception_message])
    exception_class_name = exception.__class__.__name__

    error_code, error_type, exception_target = _get_error_details(exception, logger)
    # traceback_message = message
    traceback_obj = exception.__traceback__ if hasattr(exception, "__traceback__") else None
    if traceback_obj is None:
        inner_exception = getattr(exception, "inner_exception", None)
        if inner_exception and hasattr(inner_exception, "__traceback__"):
            traceback_obj = inner_exception.__traceback__
        else:
            traceback_obj = sys.exc_info()[2]
    traceback_not_available_msg = "Not available (exception was not raised but was returned directly)"
    if traceback_obj is not None:
        traceback_message = "\n".join(traceback.format_tb(traceback_obj))
    else:
        traceback_message = traceback_not_available_msg
    logger_message = "\n".join([
        "Type: {}".format(error_type),
        "Code: {}".format(error_code),
        "Class: {}".format(exception_class_name),
        "Message: {}".format(message),
        "Traceback: {}".format(traceback_message),
        "ExceptionTarget: {}".format(exception_target)
    ])

    extra = {
        "properties": {
            "error_code": error_code,
            "error_type": error_type,
            "exception_class": exception_class_name,
            "message": message,
            "exception_traceback": traceback_message,
            "exception_target": exception_target,
        },
        "exception_tb_obj": traceback_obj,
    }

    logger.error(logger_message, extra=extra)


def log_traceback(exception: Union[AzureMLException, BaseException], logger, message=None):
    """Log exceptions without PII in APP Insights and full tracebacks in logger. Calls _log_traceback.

    Args:
        exception (_type_): _description_
        logger (_type_): _description_
        message (_type_): _description_
    """
    try:
        _log_traceback(exception, logger, message)
    except Exception as traceback_exception:
        logger.error("Failed to log exception during {} failure.".format(exception.__class__.__name__))
        _log_traceback(traceback_exception, logger)


logger = get_logger(__name__)


def log_params_and_metrics(
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    log_to_parent: bool,
) -> None:
    """Log mlflow params and metrics to current run and parent run."""
    filtered_metrics = {}
    for key in metrics:
        if isinstance(metrics[key], bool):
            # For bool value, latest version of mlflow throws an error.
            filtered_metrics[key] = float(metrics[key])
        elif isinstance(metrics[key], (int, float)):
            filtered_metrics[key] = metrics[key]
    # Log to current run
    logger.info(
        f"Attempting to log {len(parameters)} parameters and {len(filtered_metrics)} metrics."
    )
    try:
        log_mlflow_params(**parameters)
    except Exception as ex:
        logger.error(f"Failed to log parameters to current run due to {ex}")
    try:
        mlflow.log_metrics(filtered_metrics)
    except Exception as ex:
        logger.error(f"Failed to log metrics to current run due to {ex}")
    if log_to_parent:
        # Log to parent run
        try:
            parent_run_id = Run.get_context().parent.id
            ml_client = mlflow.tracking.MlflowClient()
            for param_name, param_value in parameters.items():
                param_value_to_log = param_value
                if isinstance(param_value, str) and len(param_value) > 500:
                    param_value_to_log = param_value[: 497] + '...'
                try:
                    ml_client.log_param(parent_run_id, param_name, param_value_to_log)
                except Exception as ex:
                    logger.error(f"Failed to log parameter {param_name} to root run due to {ex}.")
            for metric_name, metric_value in filtered_metrics.items():
                try:
                    ml_client.log_metric(parent_run_id, metric_name, metric_value)
                except Exception as ex:
                    logger.error(f"Failed to log metric {metric_name} to root run due to {ex}.")
        except Exception as ex:
            logger.error(f"Failed to log parameters and metrics to root run due to {ex}.")
