# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score minibatch started event."""

from src.batch_score.common.telemetry.events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent

from tests.fixtures.configuration import TEST_SCORING_URI
from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
)


def test_init(mock_run_context, make_batch_score_minibatch_started_event):
    """Test init function."""
    # Arrange & Act
    result: BatchScoreMinibatchStartedEvent = make_batch_score_minibatch_started_event

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == '2'
    assert result.scoring_url == TEST_SCORING_URI
    assert result.batch_pool == "test_pool"
    assert result.quota_audience == "test_audience"
    assert result.input_row_count == 10
