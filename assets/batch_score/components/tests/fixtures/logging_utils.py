# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock logging utilities."""

import pytest
from mock import MagicMock


@pytest.fixture()
def mock_get_logger(monkeypatch):
    """Get mock logger."""
    mock_logger = MagicMock()

    def _get_logger():
        return mock_logger

    monkeypatch.setattr("src.batch_score.common.telemetry.logging_utils.get_logger", _get_logger)
    return mock_logger


@pytest.fixture()
def mock_get_events_client(monkeypatch):
    """Get mock events client."""
    ret = MagicMock()

    def _get_events_client():
        return ret

    monkeypatch.setattr("src.batch_score.common.telemetry.logging_utils.get_events_client", _get_events_client)
    return ret
