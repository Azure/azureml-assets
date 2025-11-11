import time
import os
from foundation.model.serve.logging_config import configure_logger

logger = configure_logger(__name__)

def log_execution_time(func):
    """Decorate a function to log the execution time.

    :param func: The function to be decorated.
    :return: The decorated function.
    """

    def wrapper(*args, **kwargs):
        """Calculate and log the execution time.

        :param args: Positional arguments for the decorated function.
        :param kwargs: Keyword arguments for the decorated function.
        :return: The result of the decorated function.
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
    """Log a message, but in a box."""
    row = len(message)
    h = "".join(["+"] + ["-" * row] + ["+"])
    result = "\n" + h + "\n" + message + "\n" + h
    logger.info(result)