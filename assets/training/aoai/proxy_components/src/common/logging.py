# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging Utils."""

import logging
import sys
from importlib.metadata import entry_points

AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT = "azureml-benchmark-custom-logger"


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
    return logger
