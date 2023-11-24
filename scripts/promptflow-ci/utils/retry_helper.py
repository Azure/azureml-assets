import requests
import time
from functools import wraps
from .logging_utils import log_debug, log_error, log_warning


def retry_helper(retry_count=3):
    def retry(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            sleep_time = [1, 2, 4]
            if retry_count > 3:
                sleep_time.extend([10 for i in range(3, retry_count)])
            for i in range(retry_count):
                try:
                    result = func(*args, **kwargs)
                    if result is None:
                        log_debug(f"{func.__name__} returns None, sleep {sleep_time[i]}s and will retry for {i + 1} "
                                  f"attempt")
                        time.sleep(sleep_time[i])
                    else:
                        return result
                except Exception as e:
                    # do not retry for 401, 403
                    if (e is requests.exceptions.HTTPError and e.response.status_code in [401, 403]) or \
                            i == retry_count - 1:
                        log_error(
                            f"{func.__name__} failed after {retry_count} retry attempts. Error: {e}")
                        raise e
                    else:
                        log_warning(f"{func.__name__} failed, will sleep {sleep_time[i]} seconds and retry for the "
                                    f"{i + 1} attempt. Error: {e}")
                        time.sleep(sleep_time[i])

            result = func(*args, **kwargs)
            assert result is not None, f"Failed to {func.__name__}, which returns None"
            return result

        return wrapper

    return retry
