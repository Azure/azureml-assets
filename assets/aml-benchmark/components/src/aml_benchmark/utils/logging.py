# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for logging."""

from typing import Any
import logging
import os
import sys
import traceback
from collections import deque
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
