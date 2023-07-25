# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for the benchmarking components."""

from typing import Any
import logging
import os
import hashlib
import mlflow


def log_mlflow_params(**kwargs: Any) -> None:
    """
    Log the provided key-value pairs as parameters in MLflow.

    If a file path is provided in the value, the checksum of the file is
    calculated and logged as the parameter value. If `None` is provided as
    the value, the parameter is not logged.

    :param **kwargs: Key-value pairs of parameters to be logged in MLflow.
    :return: None
    """
    params = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and os.path.isfile(value):
            # calculate checksum of input dataset
            checksum = hashlib.md5(open(value, "rb").read()).hexdigest()
            params[key] = checksum
        else:
            if value is not None:
                params[key] = value

    mlflow.log_params(params)


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
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    return logger
