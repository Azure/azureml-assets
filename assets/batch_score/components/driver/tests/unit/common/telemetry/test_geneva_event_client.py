# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for geneva event client."""

from src.batch_score.common.telemetry.geneva_event_client import GenevaEventClient
from src.batch_score.common.telemetry.required_fields import RequiredFields
from src.batch_score.common.telemetry.standard_fields import StandardFields
from src.batch_score.common.telemetry.events.batch_score_init_completed_event import BatchScoreInitCompletedEvent
from src.batch_score.common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent
from src.batch_score.common.telemetry.events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from src.batch_score.common.telemetry.events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent

from tests.fixtures.configuration import TEST_COMPONENT_NAME, TEST_COMPONENT_VERSION

def test_generate_required_fields(mock_import_module, make_batch_score_init_completed_event):
    # Arrange
    test_event: BatchScoreInitCompletedEvent = make_batch_score_init_completed_event

    # Act
    result: RequiredFields = GenevaEventClient().generate_required_fields(test_event)

    # Assert
    assert result.SubscriptionId == "00000000-0000-0000-0000-000000000000"
    assert result.WorkspaceId == "11111111-1111-1111-111111111111"
    assert result.EventName == "BatchScore.Init.Completed"

def test_generate_standard_fields(mock_import_module, make_batch_score_init_completed_event):
    # Arrange
    test_event: BatchScoreInitCompletedEvent = make_batch_score_init_completed_event

    # Act
    result: StandardFields = GenevaEventClient().generate_standard_fields(test_event)

    # Assert
    assert result.WorkspaceRegion == "eastus"
    assert result.ClientVersion == TEST_COMPONENT_VERSION
    assert result.RunId == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert result.ParentRunId == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert result.ExperimentId == "22222222-2222-2222-2222-222222222222"

def test_generate_extension_fields_init_completed(mock_import_module, make_batch_score_init_completed_event):
    # Arrange
    test_event: BatchScoreInitCompletedEvent = make_batch_score_init_completed_event

    # Act
    result = GenevaEventClient().generate_extension_fields(test_event)

    # Assert
    assert result == {
        # Common fields
        'api_type': 'chat_completion',
        'async_mode': False,
        'authentication_type': 'api_key',
        'component_name': f'{TEST_COMPONENT_NAME}',
        'component_version': f'{TEST_COMPONENT_VERSION}',
        'endpoint_type': 'AOAI',
        'event_time': '2024-01-01T08:30:00.123456+00:00',
        'execution_mode': 'aml_pipeline',
        'resource_group': 'testrg',
        'workspace_name': 'testws',

        # Event specific fields
        'init_duration_ms': 5,
    }

def test_generate_extension_fields_init_started(mock_import_module, make_batch_score_init_started_event):
    # Arrange
    test_event: BatchScoreInitStartedEvent = make_batch_score_init_started_event

    # Act
    result = GenevaEventClient().generate_extension_fields(test_event)

    # Assert
    assert result == {
        # Common fields
        'api_type': 'chat_completion',
        'async_mode': False,
        'authentication_type': 'api_key',
        'component_name': f'{TEST_COMPONENT_NAME}',
        'component_version': f'{TEST_COMPONENT_VERSION}',
        'endpoint_type': 'AOAI',
        'event_time': '2024-01-01T08:30:00.123456+00:00',
        'execution_mode': 'aml_pipeline',
        'resource_group': 'testrg',
        'workspace_name': 'testws',
    }

def test_generate_extension_fields_minibatch_completed(mock_import_module, make_batch_score_minibatch_completed_event):
    # Arrange
    test_event: BatchScoreMinibatchCompletedEvent = make_batch_score_minibatch_completed_event

    # Act
    result = GenevaEventClient().generate_extension_fields(test_event)

    # Assert
    assert result == {
        # Common fields
        'api_type': 'chat_completion',
        'async_mode': False,
        'authentication_type': 'api_key',
        'component_name': f'{TEST_COMPONENT_NAME}',
        'component_version': f'{TEST_COMPONENT_VERSION}',
        'endpoint_type': 'AOAI',
        'event_time': '2024-01-01T08:30:00.123456+00:00',
        'execution_mode': 'aml_pipeline',
        'resource_group': 'testrg',
        'workspace_name': 'testws',

        # Event specific fields
        'minibatch_id': '2',
        'scoring_url': 'https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview',
        'batch_pool': 'test_pool',
        'quota_audience': 'test_audience',

        'total_prompt_tokens': 50,
        'total_completion_tokens': 1000,

        'input_row_count': 10,
        'output_row_count': 8,

        'http_request_count': 10,
        'http_request_succeeded_count': 5,
        'http_request_user_error_count': 3,
        'http_request_system_error_count': 2,
        'http_request_retry_count': 40,

        'http_request_duration_p0_ms': 0,
        'http_request_duration_p50_ms': 2,
        'http_request_duration_p90_ms': 5,
        'http_request_duration_p95_ms': 7,
        'http_request_duration_p99_ms': 10,
        'http_request_duration_p100_ms': 30,

        'progress_duration_p0_ms': 100,
        'progress_duration_p50_ms': 102,
        'progress_duration_p90_ms': 105,
        'progress_duration_p95_ms': 107,
        'progress_duration_p99_ms': 110,
        'progress_duration_p100_ms': 130,
    }

def test_generate_extension_fields_minibatch_started(mock_import_module, make_batch_score_minibatch_started_event):
    # Arrange
    test_event: BatchScoreMinibatchStartedEvent = make_batch_score_minibatch_started_event

    # Act
    result = GenevaEventClient().generate_extension_fields(test_event)

    # Assert
    assert result == {
        # Common fields
        'api_type': 'chat_completion',
        'async_mode': False,
        'authentication_type': 'api_key',
        'component_name': f'{TEST_COMPONENT_NAME}',
        'component_version': f'{TEST_COMPONENT_VERSION}',
        'endpoint_type': 'AOAI',
        'event_time': '2024-01-01T08:30:00.123456+00:00',
        'execution_mode': 'aml_pipeline',
        'resource_group': 'testrg',
        'workspace_name': 'testws',

        # Event specific fields
        'minibatch_id': '2',
        'scoring_url': 'https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview',
        'batch_pool': 'test_pool',
        'quota_audience': 'test_audience',
        'input_row_count': 10,
    }
