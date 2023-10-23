# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utitilies."""
from azureml.telemetry.logging_handler import get_appinsights_log_handler
from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from azureml.metrics.common.exceptions import MetricsException
from constants import TelemetryConstants
from error_definitions import ModelEvaluationInternalError
from run_utils import TestRun

from typing import Tuple, Union
from functools import wraps
import platform
import constants
import uuid
import json
import azureml.core
import traceback
import sys
import time
import logging


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

    if isinstance(exception, (AzureMLException, MetricsException)):
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


def _log_traceback(exception: (AzureMLException, BaseException), logger, message=None):
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


def log_traceback(exception: (AzureMLException, BaseException), logger, message=None):
    try:
        _log_traceback(exception, logger, message)
    except Exception as traceback_exception:
        logger.error("Failed to log exception during {} failure.".format(exception.__class__.__name__))
        _log_traceback(traceback_exception, logger)


def get_azureml_exception(exception_cls, error_cls, exception, target=None, wrap_azureml_ex=True, **message_kwargs):
    if not wrap_azureml_ex and isinstance(exception, (AzureMLException, MetricsException)):
        azureml_exception = exception
    else:
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
