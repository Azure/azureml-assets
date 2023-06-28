"""Logging Utilities."""
import logging
import constants
import json
import traceback
from azureml.rag.utils.logging import (
    get_logger,
    _logger_factory,
    enable_appinsights_logging,
    enable_stdout_logging,
    track_info,
    default_custom_dimensions,
    ActivityLoggerAdapter
)
from azureml.exceptions import AzureMLException
from typing import Union, Tuple

enable_appinsights_logging()
enable_stdout_logging()

logger = get_logger("autoprompt_logging_utilities")


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
    error_code = constants.ErrorTypes.Unclassified
    error_type = constants.ErrorTypes.Unclassified
    exception_target = default_target
    if isinstance(exception, AzureMLException):
        try:
            serialized_ex = json.loads(exception._serialize_json())
            error = serialized_ex.get(
                "error", {"code": constants.ErrorTypes.Unclassified, "inner_error": {}, "target": default_target}
            )

            # This would be the complete hierarchy of the error
            error_code = str(error.get("inner_error", constants.ErrorTypes.Unclassified))

            # This is one of 'UserError' or 'SystemError'
            error_type = error.get("code")

            exception_target = error.get("target")
            return error_code, error_type, exception_target
        except Exception:
            logger.warning(
                "Failed to parse error details while logging traceback from exception of type {}".format(exception)
            )
    return error_code, error_type, exception_target


def log_traceback(exception: AzureMLException, logger, custom_dimensions):
    """Log exceptions without PII in APP Insights and full tracebacks in logger.

    Args:
        exception (_type_): _description_
        logger (_type_): _description_
        message (_type_): _description_
        is_critical (bool, optional): _description_. Defaults to False.
    """
    exception_class_name = exception.__class__.__name__

    error_code, error_type, exception_target = _get_error_details(exception, logger)
    traceback_obj = exception.__traceback__
    traceback_message = exception.message
    if traceback_obj is None:
        if getattr(traceback_obj, "inner_exception", None):
            traceback_obj = exception.inner_exception.__traceback__
    if traceback_obj is not None:
        traceback_message = "\n".join(traceback.format_tb(traceback_obj))
    message = [
        "Type: {}".format(error_type),
        "Code: {}".format(error_code),
        "Class: {}".format(exception_class_name),
        "Message: {}".format(exception.message),
        "Traceback: {}".format(traceback_message),
        "ExceptionTarget: {}".format(exception_target)
    ]
    track_info(logger, "\n".join(message), custom_dimensions)
    _logger_factory.appinsights.flush()


def log_info(logger, message, custom_dimensions):
    """Log info to appinsights."""
    track_info(logger, message, custom_dimensions)


def log_warning(logger, message, custom_dimensions):
    """Log warning to appinsights."""
    if _logger_factory.appinsights:
        payload = {}
        payload.update(custom_dimensions)
        payload.update(default_custom_dimensions)
        child_logger = logger.getChild('appinsights')
        child_logger.addHandler(_logger_factory.appinsights)
        if ActivityLoggerAdapter:
            activity_logger = ActivityLoggerAdapter(child_logger, payload)
            activity_logger.warning(message)
