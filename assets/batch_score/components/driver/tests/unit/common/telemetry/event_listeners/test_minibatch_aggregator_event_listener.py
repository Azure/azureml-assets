# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from unittest.mock import call, MagicMock

from src.batch_score.common.telemetry.event_listeners import minibatch_aggregator_event_listener
from src.batch_score.common.telemetry.event_listeners.minibatch_aggregator_event_listener import (
    setup_minibatch_aggregator_event_handlers,
    teardown_minibatch_aggregator_event_handlers,
)
from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_input_row_completed_event import (
    BatchScoreInputRowCompletedEvent
)


def test_handle_batch_score_event(mock_run_context):
    # Arrange
    minibatch_aggregator = MagicMock()
    minibatch_aggregator_event_listener._minibatch_aggregator = minibatch_aggregator

    input_row_completed_event = BatchScoreInputRowCompletedEvent()
    my_minibatch_id = "my_minibatch_id"
    kwargs = {
        "minibatch_id": my_minibatch_id,
        "timestamp": datetime.now(),
        "output_row_count": 100,
    }

    # Setup
    setup_minibatch_aggregator_event_handlers()

    # Act
    event_utils.emit_event(batch_score_event=input_row_completed_event)
    event_utils.generate_minibatch_summary(**kwargs)

    # Assert
    minibatch_aggregator.add.assert_has_calls([
        call(input_row_completed_event),
    ])
    minibatch_aggregator.summarize.assert_called_with(**kwargs)
    minibatch_aggregator.clear.assert_called_with(minibatch_id=my_minibatch_id)

    # Teardown
    minibatch_aggregator.reset_mock()
    teardown_minibatch_aggregator_event_handlers()

    # Act
    event_utils.emit_event(batch_score_event=input_row_completed_event)
    event_utils.generate_minibatch_summary(**kwargs)

    # Assert
    minibatch_aggregator.add.assert_not_called()
    minibatch_aggregator.summarize.assert_not_called()
    minibatch_aggregator.clear.assert_not_called()
