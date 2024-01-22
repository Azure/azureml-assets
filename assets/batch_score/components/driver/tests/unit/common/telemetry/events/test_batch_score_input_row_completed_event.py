# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score input row completed event."""

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_input_row_completed_event import (
    BatchScoreInputRowCompletedEvent
)

from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
    update_common_fields,
)


def test_init(mock_run_context, make_configuration, make_metadata):
    # Arrange
    configuration: Configuration = make_configuration
    event_utils.setup_context_vars(configuration, make_metadata)

    # Act
    result = BatchScoreInputRowCompletedEvent(
        minibatch_id='test_minibatch',
        input_row_id='test_input_row',
        worker_id='5',
        scoring_url=configuration.scoring_url,
        is_successful=False,
        response_code=500,
        prompt_tokens=50,
        completion_tokens=None,
        retry_count=3,
        duration_ms=8,
        segment_count=5
    )
    update_common_fields(result)

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == 'test_minibatch'
    assert result.input_row_id == 'test_input_row'
    assert result.worker_id == '5'
    assert result.scoring_url == configuration.scoring_url
    assert result.is_successful is False
    assert result.response_code == 500
    assert result.prompt_tokens == 50
    assert result.completion_tokens is None
    assert result.retry_count == 3
    assert result.duration_ms == 8
    assert result.segment_count == 5
