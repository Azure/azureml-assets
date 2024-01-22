# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for batch score init started event."""

from src.batch_score.common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent

from tests.fixtures.telemetry_events import (
    assert_common_fields,
    assert_run_context_fields,
    TEST_COMPONENT_NAME,
    TEST_COMPONENT_VERSION,
)

def test_init(mock_run_context, make_batch_score_init_started_event):
    # Arrange & Act
    result: BatchScoreInitStartedEvent = make_batch_score_init_started_event

    # Assert
    assert_common_fields(result)
    assert_run_context_fields(result)

def test_str(mock_run_context, make_batch_score_init_started_event):
    # Arrange
    event: BatchScoreInitStartedEvent = make_batch_score_init_started_event

    # Act
    result = event.__str__()

    # Assert
    expected_texts = [
        'event_time',
        'endpoint_type: AOAI',
        'authentication_type: api_key',
        'api_type: chat_completion',
        f'component_name: {TEST_COMPONENT_NAME}',
        f'component_version: {TEST_COMPONENT_VERSION}',
    ]

    assert all(text in result for text in expected_texts)
