# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definitions for timeout utils."""

import os
import aiohttp

from ..common import constants


MINIMUM_SCORING_TIMEOUT = 10

def get_next_retry_timeout(timeout_generator):
    """Get next retry timeout."""
    try:
        return next(timeout_generator)
    except StopIteration:
        # We may encounter this if there is no max_retry_time_interval configured to stop attempting to
        # process the queue item when scoring duration continues to grow. In that case, the customer
        # wants to allow the score to retry forever. Setting the timeout to None will default to use the
        # aiohttp.ClientSession's timeout, which is set in `Conductor.__configure_session_timeout`
        return None


def get_retry_timeout_generator(default_timeout: aiohttp.ClientTimeout):
    """Get a Python generator that yields aiohttp.ClientTimeout objects."""
    for iteration in range(2, 10):
        timeout = max(_get_initial_request_timeout(), int(2 ** iteration))
        if timeout >= default_timeout.total:
            break
        else:
            yield aiohttp.ClientTimeout(timeout)
    yield default_timeout


def _get_initial_request_timeout():
    return int(os.environ.get(constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR)
               or MINIMUM_SCORING_TIMEOUT)