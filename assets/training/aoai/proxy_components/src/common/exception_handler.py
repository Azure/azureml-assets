# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exception handler class."""
import requests
import time
import openai as oai
from functools import wraps
from common.logging import get_logger
logger = get_logger(__name__)


MAX_RETRY_FOR_RETRIABLE_EXCEPTIONS = 5
MAX_RETRY_FOR_NON_RETRIABLE_EXCEPTIONS = 2
BASE_RETRY_DELAY_SEC = 10
MAX_RETRY_DELAY_SEC = 120


def _is_retriable(exception: Exception) -> bool:
    if isinstance(exception, (oai.APIConnectionError,
                              oai.APITimeoutError,
                              oai.ConflictError,
                              oai.RateLimitError)):
        return True
    if isinstance(exception, (oai.BadRequestError,
                              oai.AuthenticationError,
                              oai.PermissionDeniedError,
                              oai.UnprocessableEntityError,
                              oai.UnprocessableEntityError,
                              oai.InternalServerError)):
        return False

    if isinstance(exception, (TimeoutError, requests.Timeout, ConnectionError, requests.ConnectionError)):
        return True

    if isinstance(exception, (ValueError, TypeError, SyntaxError, AttributeError)):
        return False

    if isinstance(exception, requests.HTTPError):
        # Handle specific HTTP error status codes
        if exception.response.status_code in [500, 502, 503, 504, 429]:
            return True
        elif exception.response.status_code in [400, 401, 403]:
            # Bad request (400), unauthorized (401), forbidden (403) are non-retriable
            return False

    return False


def _get_retry_delay_seconds(attempt_count=0) -> int:
    delay = BASE_RETRY_DELAY_SEC*(2 ** (attempt_count-1))
    return min(delay, MAX_RETRY_DELAY_SEC)


def retry_on_exception(func):
    """Retry on exception."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        attempt = 0
        while True:
            try:
                attempt += 1
                return func(*args, **kwargs)
            except Exception as e:
                if _is_retriable(e):
                    logger.error(f"Retriable exception occurred: {e}. type: {type(e)},\
                                 attempt: {attempt}", exc_info=True)
                    if attempt > MAX_RETRY_FOR_RETRIABLE_EXCEPTIONS:
                        logger.error("max retries exceeded, Raising the exception")
                        raise
                else:
                    logger.error(f"Non Retriable exception occurred: {e}. type: {type(e)},\
                                 attempt: {attempt}", exc_info=True)
                    if attempt > MAX_RETRY_FOR_NON_RETRIABLE_EXCEPTIONS:
                        logger.error("max retries exceeded, Raising the exception")
                        raise

                delay_sec = _get_retry_delay_seconds(attempt)
                logger.error(f"Retrying after {delay_sec} second of delay.")
                time.sleep(delay_sec)  # Delay between retries
    return wrapper
