# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Utility functions for logging and performance monitoring.

This module provides decorators and utilities for logging function execution times
and formatting log messages.
"""
import time
import os
from foundation.model.serve.logging_config import configure_logger

logger = configure_logger(__name__)


def log_execution_time(func):
    """Decorate a function to log its execution time.

    Args:
        func: The function to be decorated.
        
    Returns:
        function: The decorated function that logs execution time.
    """

    def wrapper(*args, **kwargs):
        """Calculate and log the execution time of the decorated function.

        Args:
            *args: Positional arguments for the decorated function.
            **kwargs: Keyword arguments for the decorated function.
            
        Returns:
            The result of the decorated function.
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        if func.__name__ == "wait_until_server_healthy" and os.environ.get("LOGGING_WORKER_ID", "") == str(
            os.getpid(),
        ):
            logger.info(
                f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute.",
            )
        return result

    return wrapper


def box_logger(message: str):
    """Log a message with a decorative box border.
    
    Args:
        message (str): The message to log in a box.
    """
    row = len(message)
    h = "".join(["+"] + ["-" * row] + ["+"])
    result = "\n" + h + "\n" + message + "\n" + h
    logger.info(result)
