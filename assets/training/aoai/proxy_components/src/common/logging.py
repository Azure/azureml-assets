# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utils."""

import logging
import sys
from importlib.metadata import entry_points
from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.telemetry.logging_handler import get_appinsights_log_handler, AppInsightsLoggingHandler

AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT = "azureml-benchmark-custom-logger"

# Logs with following string in them will not be sent to appinsights
LOGS_TO_BE_FILTERED_APPINSIGHTS = [
    '[logging_handler]'
]

def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger based on the provided filename.

    This function creates a logger with the specified filename and configures it
    by setting the logging level to INFO, adding a StreamHandler to the logger,
    and specifying a specific log message format.

    :param filename: The name of the file associated with the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stream_handler)

    for custom_logger in entry_points(group=AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT):
        logger.addHandler(custom_logger.load())

    formatter = logging.Formatter(
        "[%(asctime)s - %(module)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    _add_application_insights_handler(logger)
    return logger

def _add_application_insights_handler(logger: logging.Logger):
    appinsights_handler = get_appinsights_log_handler(INSTRUMENTATION_KEY, logger=logger)
    formatter = logging.Formatter("[%(module)s] - %(message)s")
    appinsights_handler.setFormatter(formatter)
    appinsights_handler.setLevel(logging.DEBUG)
    appinsights_handler._synchronous_client.add_telemetry_processor(_appinsights_filter_processor)
    appinsights_handler._default_client.add_telemetry_processor(_appinsights_filter_processor)
    logger.addHandler(appinsights_handler)

def add_custom_dimenions_to_app_insights_handler(logger: logging.Logger,  endpoint_name, endpoint_resource_group, endpoint_subscription):
    properties = { "endpoint_name" : endpoint_name, "endpoint_resource_group": endpoint_resource_group, "endpoint_subscription": endpoint_subscription }
    for handler in logger.handlers:
        if isinstance(handler, AppInsightsLoggingHandler):
            handler._default_client.context.properties.update(properties)

def _appinsights_filter_processor(data, context) -> bool:
    """
    A process that will be added to TelemetryClient that will prevent any PII debug/info/warning from getting logged
    """
    # Do not log statements to be filtered
    if data.message is not None:
        data_message = data.message.lower()
        # Loop through all the preset strings and check if the current log contains any one of those strings
        if any([filter_str.lower() in data_message for filter_str in LOGS_TO_BE_FILTERED_APPINSIGHTS]):
            return False
    return True