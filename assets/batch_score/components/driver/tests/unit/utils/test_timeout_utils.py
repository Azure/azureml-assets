# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for timeout utils."""

import os
import pytest

from unittest import mock

from src.batch_score.common import constants
from src.batch_score.utils import timeout_utils


@pytest.mark.parametrize(
    "env_var_overrides, expected_timeouts",
    [
        (
            # Default values. Timeouts were clamped at 1800 (30 minutes).
            {},
            [10, 20, 40, 80, 160, 320, 640, 1280, 1800, 1800],
        ),
        (
            # No backoff. Steady retries.
            {
                constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR: '3',
                constants.BATCH_SCORE_BACKOFF_FACTOR_REQUEST_TIMEOUT_ENV_VAR: '1',
            },
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        ),
        (
            # High backoff factor. Timeouts were clamped at 1800 (30 minutes).
            {
                constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR: '1',
                constants.BATCH_SCORE_BACKOFF_FACTOR_REQUEST_TIMEOUT_ENV_VAR: '5',
            },
            [1, 5, 25, 125, 625, 1800, 1800, 1800, 1800, 1800],
        ),
        (
            # Start at 0.5 seconds, max at 10 seconds.
            {
                constants.BATCH_SCORE_INITIAL_REQUEST_TIMEOUT_ENV_VAR: '0.5',
                constants.BATCH_SCORE_MAX_REQUEST_TIMEOUT_ENV_VAR: '10',
            },
            [0.5, 1, 2, 4, 8, 10, 10, 10, 10, 10],
        )
    ])
def test_get_retry_timeout_generator(env_var_overrides, expected_timeouts):
    """Test get retry timeout generator with default values."""
    # Arrange
    timeout_generator = timeout_utils.get_retry_timeout_generator()

    # Act
    with mock.patch.dict(os.environ, env_var_overrides):
        timeouts = [next(timeout_generator) for _ in range(len(expected_timeouts))]

    # Assert
    assert timeouts == expected_timeouts


def test_get_retry_timeout_generator_with_max_exponential_factor():
    """Test get retry timeout generator with default values."""
    # Arrange
    timeout_generator = timeout_utils.get_retry_timeout_generator()

    # Act
    with mock.patch.dict(os.environ, {}):
        timeouts = [next(timeout_generator) for _ in range(260)]

    # Assert
    assert max(timeouts) <= 1800
