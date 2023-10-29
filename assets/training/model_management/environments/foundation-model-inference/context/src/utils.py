# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides a decorator to log the execution time of functions."""

import time
from logging_config import configure_logger

logger = configure_logger(__name__)


def log_execution_time(func):
    """
    Decorate a function to log the execution time.

    :param func: The function to be decorated.
    :return: The decorated function.
    """

    def wrapper(*args, **kwargs):
        """
        Calculate and log the execution time.

        :param args: Positional arguments for the decorated function.
        :param kwargs: Keyword arguments for the decorated function.
        :return: The result of the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute."
        )
        return result

    return wrapper
