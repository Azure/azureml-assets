# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for timeout utils."""

import aiohttp
import os
import pytest

from unittest import mock

from src.batch_score.common import constants
from src.batch_score.utils import timeout_utils


@pytest.mark.parametrize("time, expected_iters",
                         [(5, 1), (10, 1), (100, 6), (30*60*60, 9)])
def test_get_retry_timeout_generator_default_initial_timeout(time, expected_iters):
    """Test get retry timeout generator with default initial timeout."""
    # Arrange
    t = aiohttp.ClientTimeout(time)
    timeout_generator = timeout_utils.get_retry_timeout_generator(t)

    # Act
    for i in range(expected_iters):
        timeout = next(timeout_generator)

    with pytest.raises(StopIteration):
        next(timeout_generator)

    # Assert
    assert timeout.total == time


@pytest.mark.parametrize("time, expected_iters",
                         [(5, 1), (10, 1), (100, 6), (30*60*60, 9)])
def test_get_next_retry_timeout(time, expected_iters):
    """Test get next retry timeout."""
    # Arrange
    t = aiohttp.ClientTimeout(time)
    timeout_generator = timeout_utils.get_retry_timeout_generator(t)

    # Act
    for i in range(expected_iters):
        timeout_utils.get_next_retry_timeout(timeout_generator)

    # Assert
    assert timeout_utils.get_next_retry_timeout(timeout_generator) is None


@mock.patch.dict(os.environ, {constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR: '20'})
@pytest.mark.parametrize("time, expected_iters",
                         [(5, 1), (10, 1), (100, 6), (30*60*60, 9)])
def test_get_retry_timeout_generator_with_env_var_greater_than_default(time, expected_iters):
    """Test get retry timeout generator with env var for timeout greater than default."""
    # Arrange
    t = aiohttp.ClientTimeout(time)
    timeout_generator = timeout_utils.get_retry_timeout_generator(t)
    actual_iters = 0

    # Act & assert
    for i in range(expected_iters):
        timeout = next(timeout_generator)
        actual_iters = i + 1
        if i+1 != expected_iters:
            assert timeout.total == max(20, 2**(i+2))

    with pytest.raises(StopIteration):
        next(timeout_generator)

    assert timeout.total == time
    assert actual_iters == expected_iters


@mock.patch.dict(os.environ, {constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR: '3'})
@pytest.mark.parametrize("time, expected_iters",
                         [(5, 2), (10, 3), (100, 6), (30*60*60, 9)])
def test_get_retry_timeout_generator_with_env_var_less_than_default(time, expected_iters):
    """Test get retry timeout generator with env var for timeout less than default."""
    # Arrange
    t = aiohttp.ClientTimeout(time)
    timeout_generator = timeout_utils.get_retry_timeout_generator(t)
    actual_iters = 0

    # Act & assert
    for i in range(expected_iters):
        timeout = next(timeout_generator)
        actual_iters = i + 1
        if i+1 != expected_iters:
            assert timeout.total == max(3, 2**(i+2))

    with pytest.raises(StopIteration):
        next(timeout_generator)

    assert timeout.total == time
    assert actual_iters == expected_iters
