# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score worker decreased event."""

from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_worker_decreased_event import BatchScoreWorkerDecreasedEvent

from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
    update_common_fields,
)


def test_init(mock_run_context, make_configuration, make_metadata):
    """Test init function."""
    # Arrange
    event_utils.setup_context_vars(make_configuration, make_metadata)

    # Act
    result = BatchScoreWorkerDecreasedEvent(
        active_minibatch_count=10,
        previous_active_worker_count=3,
        current_active_worker_count=4,
        p90_wait_time_ms=2.5)
    update_common_fields(result)

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.active_minibatch_count == 10
    assert result.previous_active_worker_count == 3
    assert result.current_active_worker_count == 4
    assert result.p90_wait_time_ms == 2.5
