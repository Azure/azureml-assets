# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for logging."""

from typing import Any
import logging
import os
import hashlib
import mlflow


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


def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger based on the provided filename.

    This function creates a logger with the specified filename and configures it
    by setting the logging level to INFO, adding a StreamHandler and a FileHandler to the logger,
    and specifying a specific log message format for both handlers.

    :param filename: The name of the file associated with the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)

    # StreamHandler for stdout, temporary for debugging
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # FileHandler for logging to a file
    file_handler = logging.FileHandler(filename)
    logger.addHandler(file_handler)

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    return logger
