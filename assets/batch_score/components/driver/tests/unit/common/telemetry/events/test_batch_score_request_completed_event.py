# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score request completed event."""

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent

from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
    update_common_fields,
)


def test_init(mock_run_context, make_configuration, make_metadata):
    """Test init function."""
    # Arrange
    configuration: Configuration = make_configuration
    event_utils.setup_context_vars(configuration, make_metadata)

    # Act
    result = BatchScoreRequestCompletedEvent(
        minibatch_id='test_minibatch',
        input_row_id='test_input_row',
        x_ms_client_request_id='test_client_id',
        worker_id='5',
        segmented_request_id='2',
        scoring_url=configuration.scoring_url,
        is_successful=True,
        is_retriable=False,
        response_code=200,
        model_response_code=None,
        prompt_tokens=100,
        completion_tokens=1000,
        duration_ms=8
    )
    update_common_fields(result)

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == 'test_minibatch'
    assert result.input_row_id == 'test_input_row'
    assert result.x_ms_client_request_id == 'test_client_id'
    assert result.worker_id == '5'
    assert result.segmented_request_id == '2'
    assert result.scoring_url == configuration.scoring_url
    assert result.is_successful
    assert result.is_retriable is False
    assert result.response_code == 200
    assert result.model_response_code is None
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 1000
    assert result.duration_ms == 8
