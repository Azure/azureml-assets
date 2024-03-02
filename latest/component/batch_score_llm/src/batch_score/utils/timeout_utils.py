# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definitions for timeout utils."""

import os
from typing import Generator

from ..common import constants


BACKOFF_FACTOR_REQUEST_TIMEOUT = 2
INITIAL_REQUEST_TIMEOUT = 10
MAX_REQUEST_TIMEOUT = 30 * 60  # 30 minutes


def get_retry_timeout_generator() -> Generator[float, None, None]:
    """Get a Python generator that yields timeouts in seconds."""
    backoff_factor = _get_backoff_factor_request_timeout()
    initial_timeout = _get_initial_request_timeout()
    max_timeout = _get_max_request_timeout()

    i = 0
    while True:
        yield min(initial_timeout * (backoff_factor ** i),
                  max_timeout)
        i += 1


def _get_backoff_factor_request_timeout():
    return float(os.environ.get(constants.BATCH_SCORE_BACKOFF_FACTOR_REQUEST_TIMEOUT_ENV_VAR)
                 or BACKOFF_FACTOR_REQUEST_TIMEOUT)


def _get_initial_request_timeout():
    return float(os.environ.get(constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR)
                 or INITIAL_REQUEST_TIMEOUT)


def _get_max_request_timeout():
    return float(os.environ.get(constants.BATCH_SCORE_MAX_REQUEST_TIMEOUT_ENV_VAR)
                 or MAX_REQUEST_TIMEOUT)
