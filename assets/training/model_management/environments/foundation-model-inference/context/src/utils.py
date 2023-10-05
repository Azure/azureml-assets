import time
from logging_config import configure_logger

logger = configure_logger(__name__)


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Function {func.__name__} took {elapsed_time:.4f} seconds to execute."
        )
        return result

    return wrapper
