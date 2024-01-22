# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score request started event."""

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.events.batch_score_request_started_event import BatchScoreRequestStartedEvent

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
    result = BatchScoreRequestStartedEvent(
        minibatch_id = 'test_minibatch',
        input_row_id =  'test_input_row',
        worker_id = '5',
        segmented_request_id = '2',
        scoring_url = configuration.scoring_url
    )
    update_common_fields(result)

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

    assert result.minibatch_id == 'test_minibatch'
    assert result.input_row_id == 'test_input_row'
    assert result.worker_id == '5'
    assert result.segmented_request_id == '2'
    assert result.scoring_url == configuration.scoring_url
