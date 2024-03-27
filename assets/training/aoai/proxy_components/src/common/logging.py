# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utils."""

import logging
import sys
from importlib.metadata import entry_points
from azureml.telemetry import get_telemetry_log_handler

AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT = "azureml-benchmark-custom-logger"

# Logs with following string in them will not be sent to appinsights
LOGS_TO_BE_FILTERED_APPINSIGHTS = []

class Config:
    """
    config class
    """
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"
    INSTRUMENTATION_KEY_AML = "7b709447-0334-471a-9648-30349a41b45c"
    INSTRUMENTATION_KEY_AML_OLD = "71b954a8-6b7d-43f5-986c-3d3a6605d803"


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
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    appinsights_handler = get_application_insights_handler()
    logger.addHandler(appinsights_handler)
    return logger

def get_application_insights_handler():
     # create AppInsight handler and set formatter
    appinsights_handler = get_telemetry_log_handler(
        instrumentation_key=Config.INSTRUMENTATION_KEY_AML_OLD,
        component_name="aoai-proxy",
    )
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    appinsights_handler.setFormatter(formatter)
    appinsights_handler.setLevel(Config.VERBOSITY_LEVEL)
    appinsights_handler.set_name(Config.APP_INSIGHT_HANDLER_NAME)
    appinsights_handler._synchronous_client.add_telemetry_processor(_appinsights_filter_processor)
    appinsights_handler._default_client.add_telemetry_processor(_appinsights_filter_processor)
    return appinsights_handler


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
