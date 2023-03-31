# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utitilies."""
import logging
import platform
import constants
import uuid
import json
import azureml.core

from azureml.telemetry.logging_handler import get_appinsights_log_handler
from azureml.exceptions import AzureMLException
from run_utils import TestRun
from typing import Tuple, Union

NON_PII_MESSAGE = '[Hidden as it may contain PII]'


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

        message = properties.get('exception_message', NON_PII_MESSAGE)
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


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(self,
                 app_name=constants.TelemetryConstants.COMPONENT_NAME,
                 run_id=None,
                 common_core_version=azureml.core.__version__,
                 model_evaluation_version="0.0.5",
                 compute_target=None,
                 experiment_id=None,
                 parent_run_id=None,
                 os_info=platform.system(),
                 region=None,
                 subscription_id=None,
                 task_type="",
                 mode="") -> None:
        """__init__.

        Args:
            app_name (_type_, optional): _description_. Defaults to constants.TelemetryConstants.COMPONENT_NAME.
            run_id (_type_, optional): _description_. Defaults to None.
            common_core_version (_type_, optional): _description_. Defaults to azureml.core.__version__.
            model_evaluation_version (str, optional): _description_. Defaults to "0.0.19".
            compute_target (_type_, optional): _description_. Defaults to None.
            experiment_id (_type_, optional): _description_. Defaults to None.
            parent_run_id (_type_, optional): _description_. Defaults to None.
            os_info (_type_, optional): _description_. Defaults to platform.system().
            region (_type_, optional): _description_. Defaults to None.
            subscription_id (_type_, optional): _description_. Defaults to None.
            task_type (str, optional): _description_. Defaults to "".
            mode (str, optional): _description_. Defaults to "".
        """
        self.app_name = app_name
        self.run_id = run_id
        self.common_core_version = common_core_version
        self.model_evaluation_version = model_evaluation_version
        self.compute_target = compute_target
        self.experiment_id = experiment_id
        self.parent_run_id = parent_run_id
        self.os_info = os_info
        self.region = region
        self.subscription_id = subscription_id
        self.task_type = task_type
        self.mode = mode
        if self.run_id is None:
            run_obj = TestRun()
            self.run_id = run_obj.run.id
            self.compute_target = run_obj.compute
            self.region = run_obj.region
            self.experiment_id = run_obj.experiment.id
            self.subscription_id = run_obj.subscription
            self.parent_run_id = self.run_id
            if hasattr(run_obj.run, "parent"):
                self.parent_run_id = run_obj.run.parent.id
        if self.task_type == "":
            import sys
            args = sys.argv
            if "--task" in args:
                ind = args.index("--task")
                self.task_type = sys.argv[ind+1]
        if self.mode == "":
            import sys
            args = sys.argv
            if "--mode" in args:
                ind = args.index("--mode")
                self.mode = sys.argv[ind+1]


custom_dimensions = CustomDimensions()


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
               name: str = constants.TelemetryConstants.LOGGER_NAME):
    """Get logger.

    Args:
        logging_level (str, optional): _description_. Defaults to 'DEBUG'.
        custom_dimensions (dict, optional): _description_. Defaults to vars(custom_dimensions).
        name (str, optional): _description_. Defaults to constants.TelemetryConstants.LOGGER_NAME.

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
    app_name = constants.TelemetryConstants.COMPONENT_NAME

    if (constants.TelemetryConstants.MODEL_EVALUATION_HANDLER_NAME not in handler_names):
        formatter = logging.Formatter(
            '%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
            [%(process)d] %(message)s \n'.format(app_name, run_id)
        )
        stream_handler = ModelEvaluationHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_log_level)
        stream_handler.set_name(constants.TelemetryConstants.MODEL_EVALUATION_HANDLER_NAME)
        logger.addHandler(stream_handler)

    if (constants.TelemetryConstants.APP_INSIGHT_HANDLER_NAME not in handler_names):
        child_namespace = __name__
        current_logger = logging.getLogger("azureml.telemetry").getChild(child_namespace)
        current_logger.propagate = False
        current_logger.setLevel(logging.CRITICAL)
        appinsights_handler = get_appinsights_log_handler(
            instrumentation_key=constants.TelemetryConstants.INSTRUMENTATION_KEY,
            logger=current_logger, properties=custom_dimensions
        )
        formatter = AppInsightsPIIStrippingFormatter(
            fmt='%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
                [%(process)d] %(message)s \n'.format(app_name, run_id)
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)
        appinsights_handler.set_name(constants.TelemetryConstants.APP_INSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)

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

    error_code = constants.ExceptionTypes.Unclassified
    error_type = constants.ExceptionTypes.Unclassified
    exception_target = default_target
    return error_code, error_type, exception_target


def log_traceback(exception, logger, message, is_critical=False):
    """Log exceptions without PII in APP Insights and full tracebacks in logger.

    Args:
        exception (_type_): _description_
        logger (_type_): _description_
        message (_type_): _description_
        is_critical (bool, optional): _description_. Defaults to False.
    """
    exception_class_name = exception.__class__.__name__

    error_code, error_type, exception_target = _get_error_details(exception, logger)

    message = [
        "Type: {}".format(error_code),
        "Class: {}".format(exception_class_name),
        "Message: {}".format(message),
        "Traceback:",
        message,
        "ExceptionTarget: {}".format(exception_target),
    ]

    if is_critical:
        logger.critical("\n".join(message))
    else:
        logger.error("\n".join(message))
