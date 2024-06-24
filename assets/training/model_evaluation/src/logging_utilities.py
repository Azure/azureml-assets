# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utitilies."""
from azureml.telemetry.logging_handler import get_appinsights_log_handler
from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.automl.core.shared import log_server
from azureml.automl.core.shared.logging_utilities import (
    _CustomStackSummary, mark_package_exceptions_as_loggable, mark_path_as_loggable
)
from azureml.exceptions import AzureMLException
from azureml.metrics.common.exceptions import MetricsException
from azureml.evaluate.mlflow.exceptions import AzureMLMLFlowUserException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from azureml._common._error_definition.system_error import ClientError

from constants import TelemetryConstants
from error_definitions import ModelEvaluationInternalError, ModelPredictionUserError, ComputeMetricsUserError
from run_utils import TestRun

from typing import List, Tuple, Union, Optional
from functools import wraps
import platform
import constants
import uuid
import json
import azureml.core
import sys
from pathlib import Path
import time
import logging


def add_package_exceptions_as_loggable(packages: List):
    """Add package as loggable for Telemetry."""
    for package in packages:
        try:
            sys_module = sys.modules[package]
            # Mark package as being allowed to log certain built-in types
            mark_package_exceptions_as_loggable(sys_module)
            log_server.install_sockethandler(package)
        except Exception:
            pass


add_package_exceptions_as_loggable(TelemetryConstants.ALLOWED_EXTRA_PACKAGES)

# Mark current path as allowed
mark_path_as_loggable(str(Path(__file__).parent.absolute()))


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(self,
                 run_details,
                 app_name=TelemetryConstants.COMPONENT_NAME,
                 model_evaluation_version=TelemetryConstants.COMPONENT_DEFAULT_VERSION,
                 os_info=platform.system(),
                 task_type="") -> None:
        """__init__.

        Args:
            run_details (_type_, optional): _description_. Defaults to None.
            app_name (_type_, optional): _description_. Defaults to TelemetryConstants.COMPONENT_NAME.
            model_evaluation_version : _description_. Defaults to TelemetryConstants.COMPONENT_DEFAULT_VERSION.
            os_info (_type_, optional): _description_. Defaults to platform.system().
            task_type (str, optional): _description_. Defaults to "".
        """
        self.app_name = app_name
        self.run_id = run_details.run.id
        self.common_core_version = azureml.core.__version__
        self.compute_target = run_details.compute
        self.experiment_id = run_details.experiment.id
        self.parent_run_id = run_details.parent_run.id
        self.root_run_id = run_details.root_run.id
        self.os_info = os_info
        self.region = run_details.region
        self.subscription_id = run_details.subscription
        self.task_type = task_type
        self.rootAttribution = run_details.root_attribute
        run_info = run_details.get_extra_run_info
        self.location = run_info.get("location", "")

        self.moduleVersion = run_info.get("moduleVersion", model_evaluation_version)
        if run_info.get("moduleId", None):
            self.moduleId = run_info.get("moduleId")
        if run_info.get("moduleSource", None):
            self.moduleSource = run_info.get("moduleSource")
        if run_info.get("moduleRegistryName", None):
            self.moduleRegistryName = run_info.get("moduleRegistryName")

        if run_info.get("model_asset_id", None):
            self.model_asset_id = run_info.get("model_asset_id")
        if run_info.get("model_source", None):
            self.model_source = run_info.get("model_source")
        if run_info.get("model_registry_name", None):
            self.model_registry_name = run_info.get("model_registry_name")
        if run_info.get("model_name", None):
            self.model_name = run_info.get("model_name")
        if run_info.get("model_version", None):
            self.model_version = run_info.get("model_version")

        if run_info.get("pipeline_type", None):
            self.pipeline_type = run_info.get("pipeline_type")
        if run_info.get("source", None):
            self.source = run_info.get("source")
        if self.task_type == "":
            import sys
            args = sys.argv
            if "--task" in args:
                ind = args.index("--task")
                self.task_type = sys.argv[ind + 1]


current_run = TestRun()
custom_dimensions = CustomDimensions(current_run)


class AppInsightsPIIStrippingFormatter(logging.Formatter):
    """Formatter for App Insights Logging.

    Args:
        logging (_type_): _description_
    """

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

        message = properties.get('exception_message', TelemetryConstants.NON_PII_MESSAGE)
        traceback_msg = properties.get('exception_traceback', not_available_message)

        record.message = record.msg = '\n'.join([
            'Type: {}'.format(properties.get('error_type', constants.ExceptionTypes.Unclassified)),
            'Class: {}'.format(properties.get('exception_class', not_available_message)),
            'Message: {}'.format(message),
            'Traceback: {}'.format(traceback_msg),
            'ExceptionTarget: {}'.format(properties.get('exception_target', not_available_message))
        ])

        # Update exception message and traceback in extra properties as well
        properties['exception_message'] = message

        return super().format(record)


class ModelEvaluationHandler(logging.StreamHandler):
    """Remote/Local Run Logging Handler.

    Args:
        logging (_type_): _description_
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Log record to output stream.

        Args:
            record (logging.LogRecord): _description_
        """
        new_properties = getattr(record, "properties", {})
        new_properties.update({'log_id': str(uuid.uuid4())})
        custom_dims_dict = vars(custom_dimensions)
        cust_dim_copy = custom_dims_dict.copy()
        cust_dim_copy.update(new_properties)
        setattr(record, "properties", cust_dim_copy)
        msg = self.format(record)
        if record.levelname == 'ERROR' and 'AzureMLException' not in record.message:
            setattr(record, "exception_tb_obj", "non-azureml exception raised so scrubbing")
        stream = self.stream
        stream.write(msg)


def get_logger(logging_level: str = 'DEBUG',
               custom_dimensions: dict = vars(custom_dimensions),
               name: str = TelemetryConstants.LOGGER_NAME):
    """Get logger.

    Args:
        logging_level (str, optional): _description_. Defaults to 'DEBUG'.
        custom_dimensions (dict, optional): _description_. Defaults to vars(custom_dimensions).
        name (str, optional): _description_. Defaults to TelemetryConstants.LOGGER_NAME.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    numeric_log_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % logging_level)

    logger = logging.getLogger(name)
    logger.propagate = True
    logger.setLevel(numeric_log_level)
    handler_names = [handler.get_name() for handler in logger.handlers]

    run_id = custom_dimensions["run_id"]
    app_name = TelemetryConstants.COMPONENT_NAME

    if TelemetryConstants.MODEL_EVALUATION_HANDLER_NAME not in handler_names:
        formatter = logging.Formatter(
            '%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
            [%(process)d] %(message)s \n'.format(app_name, run_id)
        )
        stream_handler = ModelEvaluationHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_log_level)
        stream_handler.set_name(TelemetryConstants.MODEL_EVALUATION_HANDLER_NAME)
        logger.addHandler(stream_handler)

    if TelemetryConstants.APP_INSIGHT_HANDLER_NAME not in handler_names:
        child_namespace = __name__
        current_logger = logging.getLogger("azureml.telemetry").getChild(child_namespace)
        current_logger.propagate = False
        current_logger.setLevel(logging.CRITICAL)
        appinsights_handler = get_appinsights_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY,
            logger=current_logger, properties=custom_dimensions
        )
        formatter = AppInsightsPIIStrippingFormatter(
            fmt='%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
                [%(process)d] %(message)s \n'.format(app_name, run_id)
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)
        appinsights_handler.set_name(TelemetryConstants.APP_INSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)

    return logger


def flush_logger(logger):
    """Flush logger."""
    for handler in logger.handlers:
        handler.flush()
    time.sleep(20)


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
    error_code = constants.ExceptionTypes.Unclassified
    error_type = constants.ExceptionTypes.Unclassified
    exception_target = default_target

    if isinstance(exception, AzureMLException):
        try:
            serialized_ex = json.loads(exception._serialize_json())
            error = serialized_ex.get(
                "error", {"code": constants.ExceptionTypes.Unclassified, "inner_error": {}, "target": default_target}
            )

            # This would be the complete hierarchy of the error
            error_code = str(error.get("inner_error", constants.ExceptionTypes.Unclassified))

            # This is one of 'UserError' or 'SystemError'
            error_type = error.get("code")

            exception_target = error.get("target")
            return error_code, error_type, exception_target
        except Exception:
            logger.warning(
                "Failed to parse error details while logging traceback from exception of type {}".format(exception)
            )

    return error_code, error_type, exception_target


def get_pii_free_msg(exception: AzureMLException, scrubbed: bool = True) -> str:
    """
    Fallback message to use for situations where printing PII-containing information is inappropriate.

    :param scrubbed: If true, return a generic '[Hidden as it may contain PII]' as a fallback, else an empty string
    :return: Log safe message for logging in telemetry
    """
    if exception._azureml_error is not None:
        return exception._azureml_error.log_safe_message_format()  # type: str

    fallback_message = (getattr(exception, '_message_format', None) or getattr(exception, '_generic_msg', None)
                        or TelemetryConstants.NON_PII_MESSAGE if scrubbed else '')  # type: str
    has_pii = getattr(exception, '_generic_msg', None)
    message = exception._exception_message or fallback_message if not has_pii else fallback_message  # type: str
    return message


def _exception_msg_format(
        exception, error_name: str, message: str, error_response: Optional[str], log_safe: bool = True) -> str:
    inner_exception_message = None
    if exception._inner_exception:
        if log_safe:
            # Only print the inner exception type for a log safe message
            inner_exception_message = exception._inner_exception.__class__.__name__
        else:
            inner_exception_message = "{}: {}".format(
                exception._inner_exception.__class__.__name__,
                str(exception._inner_exception)
            )
    return "{}:\n\tMessage: {}\n\tInnerException: {}\n\tErrorResponse \n{}".format(
        error_name,
        message,
        inner_exception_message,
        error_response)


def get_pii_free_exception_msg_format(exception: AzureMLException) -> str:
    """Get PII free exception message format.

    :return: PII free exception message format
    """
    # Update exception message to be PII free
    # Update inner exception to log exception type only
    # Update Error Response to contain PII free message
    pii_free_msg = get_pii_free_msg(exception)
    error_dict = json.loads(exception._serialize_json(
        indent=4, filter_fields=[AzureMLError.Keys.MESSAGE_FORMAT, AzureMLError.Keys.MESSAGE_PARAMETERS]
    ))
    error_dict['error']['message'] = pii_free_msg
    return _exception_msg_format(
        exception,
        exception.__class__.__name__,
        pii_free_msg,
        json.dumps(error_dict, indent=4)
    )


def _get_pii_free_message(exception: BaseException) -> str:
    if isinstance(exception, AzureMLException):
        return get_pii_free_exception_msg_format(exception)
    else:
        return TelemetryConstants.NON_PII_MESSAGE


def _log_traceback(
        exception: (AzureMLException, BaseException),
        logger,
        override_error_msg: Optional[str] = None,
):
    """Log exceptions without PII in APP Insights and full tracebacks in logger.

    Args:
        exception (_type_): _description_
        logger (_type_): _description_
        override_error_msg (_type_): _description_
    """
    error_msg = "No message available."
    if hasattr(exception, "message"):
        error_msg = exception.message
    elif hasattr(exception, "exception_message"):
        error_msg = exception.exception_message
    error_msg = error_msg if override_error_msg is None else "\n".join([override_error_msg, error_msg])

    error_code, error_type, exception_target = _get_error_details(exception, logger)

    # Some exceptions may not have a __traceback__ attr
    traceback_obj = exception.__traceback__ if hasattr(exception, "__traceback__") else None or sys.exc_info()[2]

    traceback_msg = _CustomStackSummary.get_traceback_message(traceback_obj, remove_pii=False)

    exception_class_name = exception.__class__.__name__

    logger_message = "\n".join([
        "Type: {}".format(error_type),
        "Code: {}".format(error_code),
        "Class: {}".format(exception_class_name),
        "Message: {}".format(error_msg),
        "Traceback: {}".format(traceback_msg),
        "ExceptionTarget: {}".format(exception_target)
    ])

    # Marking extra properties to be PII free since azureml-telemetry logging_handler is
    # not updating the extra properties after formatting.
    # Get PII free exception_message
    error_msg_without_pii = _get_pii_free_message(exception)
    # Get PII free exception_traceback
    traceback_msg_without_pii = _CustomStackSummary.get_traceback_message(traceback_obj)
    # Get PII free exception_traceback

    extra = {
        "properties": {
            "error_code": error_code,
            "error_type": error_type,
            "exception_class": exception_class_name,
            "exception_message": error_msg_without_pii,
            "exception_traceback": traceback_msg_without_pii,
            "exception_target": exception_target,
        },
        "exception_tb_obj": traceback_obj,
    }

    logger.error(logger_message, extra=extra)


def log_traceback(exception: (AzureMLException, BaseException), logger, message=None):
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


def get_azureml_exception(exception_cls, error_cls, exception, target=None, wrap_azureml_ex=True, **message_kwargs):
    """Get azureml wrapped exception object.

    Args:
        exception_cls (_type_): _description_
        error_cls (_type_): _description_
        exception (_type_): _description_
        target (_type_): _description_
        wrap_azureml_ex (_type_): _description_
        message_kwargs (_type_): _description_
    """
    if not wrap_azureml_ex and isinstance(exception, AzureMLException):
        azureml_exception = exception
    else:
        if isinstance(exception, MetricsException) and isinstance(error_cls(), ClientError):
            error_cls = ComputeMetricsUserError
        elif isinstance(exception, AzureMLMLFlowUserException):
            error_cls = ModelPredictionUserError
        azureml_error = AzureMLError.create(error_cls, target=target, **message_kwargs)

        tb_obj = None
        if exception:
            tb_obj = exception.__traceback__ if hasattr(exception, "__traceback__") else sys.exc_info()[2]

        azureml_exception = exception_cls._with_error(
            azureml_error, inner_exception=exception).with_traceback(tb_obj)
    return azureml_exception


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
                exception = get_azureml_exception(AzureMLException, ModelEvaluationInternalError, e,
                                                  wrap_azureml_ex=False, error=repr(e))
                log_traceback(exception, logger)

                flush_logger(logger)
                raise exception
            finally:
                flush_logger(logger)
        return wrapper
    return wrap
