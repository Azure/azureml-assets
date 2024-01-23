# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for event utils."""

import os
import pytest

from unittest.mock import patch
from src.batch_score.common.constants import BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR
from src.batch_score.common.telemetry.events.event_utils import catch_and_log_all_exceptions


@catch_and_log_all_exceptions
def sample_method(x):
    return 10/x


def test_no_exception_thrown():
    assert sample_method(2) == 5


@patch.dict(os.environ, {BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR: 'False'}, clear=True)
def test_exception_suppressed():
    assert sample_method(0) is None


@patch.dict(os.environ, {BATCH_SCORE_SURFACE_TELEMETRY_EXCEPTIONS_ENV_VAR: 'True'}, clear=True)
def test_exception_surfaced():
    with pytest.raises(ZeroDivisionError):
        sample_method(0)
