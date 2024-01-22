# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for job log event listener."""

import logging
from unittest.mock import patch

from pydispatch import dispatcher

from src.batch_score.common.telemetry.event_listeners.job_log_event_listener import setup_job_log_event_handlers
from src.batch_score.common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent


def test_handle_batch_score_event(mock_run_context):
    # Arrange
    setup_job_log_event_handlers()
    init_started_event = BatchScoreInitStartedEvent()

    # Act
    with patch.object(init_started_event, '__str__') as mock_str:
        with patch.object(logging.LoggerAdapter, 'info') as mock_info_logger:
            dispatcher.send(batch_score_event=init_started_event)

    # Assert
    mock_str.assert_called()
    mock_info_logger.assert_called()
