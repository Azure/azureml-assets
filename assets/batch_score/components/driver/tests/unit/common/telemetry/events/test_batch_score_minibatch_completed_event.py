# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score minibatch completed event."""

from src.batch_score.common.telemetry.events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent

from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
)

def test_init(mock_run_context, make_batch_score_minibatch_completed_event):
    # Arrange & Act
    result: BatchScoreMinibatchCompletedEvent = make_batch_score_minibatch_completed_event

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == '2'
    assert result.scoring_url == "https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview"
    assert result.batch_pool == "test_pool"
    assert result.quota_audience == "test_audience"

    assert result.total_prompt_tokens== 50
    assert result.total_completion_tokens == 1000

    assert result.input_row_count == 10
    assert result.output_row_count == 8

    assert result.http_request_count == 10
    assert result.http_request_succeeded_count == 5
    assert result.http_request_user_error_count == 3
    assert result.http_request_system_error_count == 2
    assert result.http_request_retry_count == 40

    assert result.http_request_duration_p0_ms == 0
    assert result.http_request_duration_p50_ms == 2
    assert result.http_request_duration_p90_ms == 5
    assert result.http_request_duration_p95_ms == 7
    assert result.http_request_duration_p99_ms == 10
    assert result.http_request_duration_p100_ms == 30

    assert result.progress_duration_p0_ms == 100
    assert result.progress_duration_p50_ms == 102
    assert result.progress_duration_p90_ms == 105
    assert result.progress_duration_p95_ms == 107
    assert result.progress_duration_p99_ms == 110
    assert result.progress_duration_p100_ms == 130
