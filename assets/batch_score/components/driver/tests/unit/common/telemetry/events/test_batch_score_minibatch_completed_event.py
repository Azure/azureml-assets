# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score minibatch completed event."""

from src.batch_score_oss.root.common.telemetry.events.batch_score_minibatch_completed_event import (
    BatchScoreMinibatchCompletedEvent
)

from tests.fixtures.configuration import TEST_SCORING_URI
from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_http_request_fields,
    assert_run_context_fields,
)


def test_init(mock_run_context, make_batch_score_minibatch_completed_event):
    """Test init function."""
    # Arrange & Act
    result: BatchScoreMinibatchCompletedEvent = make_batch_score_minibatch_completed_event

    # Assert
    assert_common_fields(result)
    assert_http_request_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == '2'
    assert result.scoring_url == TEST_SCORING_URI
    assert result.model_name == 'test_model_name'
    assert result.retry_count == 0

    assert result.total_prompt_tokens == 50
    assert result.total_completion_tokens == 1000

    assert result.input_row_count == 10
    assert result.output_row_count == 8

    assert result.http_request_retry_count == 40

    assert result.progress_duration_p0_ms == 100
    assert result.progress_duration_p50_ms == 102
    assert result.progress_duration_p90_ms == 105
    assert result.progress_duration_p95_ms == 107
    assert result.progress_duration_p99_ms == 110
    assert result.progress_duration_p100_ms == 130
