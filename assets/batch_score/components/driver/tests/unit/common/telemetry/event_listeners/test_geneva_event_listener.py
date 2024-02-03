# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for geneva event listener."""

from unittest.mock import call, MagicMock, patch

from tests.fixtures.geneva_event_listener import mock_import
with patch('importlib.import_module', side_effect=mock_import):
    from src.batch_score.common.telemetry.event_listeners import geneva_event_listener

from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent


def test_handle_batch_score_event(mock_run_context):
    """Test handle batch score event."""
    # Arrange
    init_started_event = BatchScoreInitStartedEvent()

    # Act
    with patch.object(geneva_event_listener,
                      '_geneva_event_client',
                      new_callable=MagicMock) as mock_geneva_event_client:
        geneva_event_listener.setup_geneva_event_handlers()
        event_utils.emit_event(batch_score_event=init_started_event)

    # Assert
    mock_geneva_event_client.emit_event.assert_has_calls([
        call(init_started_event),
    ])
