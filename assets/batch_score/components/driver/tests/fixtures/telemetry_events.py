import pytest
from datetime import datetime, timezone

from src.batch_score.common.common_enums import ApiType, AuthenticationType, EndpointType
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.telemetry.events.event_utils import setup_context_vars
from src.batch_score.common.telemetry.events.batch_score_event import BatchScoreEvent
from src.batch_score.common.telemetry.events.batch_score_init_completed_event import BatchScoreInitCompletedEvent
from src.batch_score.common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent
from src.batch_score.common.telemetry.events.batch_score_minibatch_completed_event import BatchScoreMinibatchCompletedEvent
from src.batch_score.common.telemetry.events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent

from tests.fixtures.configuration import TEST_COMPONENT_NAME, TEST_COMPONENT_VERSION

TEST_EXPERIMENT_ID = "22222222-2222-2222-2222-222222222222"
TEST_REGION = "eastus"
TEST_PARENT_RUN_ID = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
TEST_RESOURCE_GROUP = "testrg"
TEST_RUN_ID = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
TEST_SUBSCRIPTION_ID = "00000000-0000-0000-0000-000000000000"
TEST_WORKSPACE_ID = "11111111-1111-1111-111111111111"
TEST_WORKSPACE_NAME = "testws"

TEST_API_TYPE = ApiType.ChatCompletion
TEST_AUTHENTICATION_TYPE = AuthenticationType.ApiKey
TEST_ENDPOINT_TYPE = EndpointType.AOAI
TEST_EVENT_TIME = datetime(2024, 1, 1, 8, 30, 0, 123456, tzinfo=timezone.utc)
TEST_EXECUTION_MODE = 'aml_pipeline'

@pytest.fixture
def mock_run_context(monkeypatch):
    class MockWorkspace():
        def __init__(self, subscription_id, resource_group, workspace_name, location, workspace_id):
            self.subscription_id = subscription_id
            self.resource_group = resource_group
            self._workspace_name = workspace_name
            self.location = location
            self._workspace_id_internal = workspace_id

    class MockExperiment():
        def __init__(self, workspace, id):
            self.workspace = workspace
            self.id = id
    
    class MockRun():
        def __init__(self, id):
            self.id = id
        
    class MockRunContext():
        def __init__(self, experiment, run_id, parent_run_id):
            self.experiment = experiment
            self._run_id = run_id
            self.parent = MockRun(parent_run_id)

    def get_mock_run_context():
        ws = MockWorkspace(
            subscription_id=TEST_SUBSCRIPTION_ID,
            resource_group=TEST_RESOURCE_GROUP,
            workspace_name=TEST_WORKSPACE_NAME,
            location=TEST_REGION,
            workspace_id=TEST_WORKSPACE_ID,
        )
        experiment = MockExperiment(workspace=ws, id=TEST_EXPERIMENT_ID)
        return MockRunContext(experiment, run_id=TEST_RUN_ID, parent_run_id=TEST_PARENT_RUN_ID)

    monkeypatch.setattr("azureml.core.Run.get_context", get_mock_run_context)

@pytest.fixture
def make_batch_score_init_completed_event(mock_run_context, make_configuration, make_metadata):
    setup_context_vars(make_configuration, make_metadata)

    return update_common_fields(BatchScoreInitCompletedEvent(init_duration_ms = 5))

@pytest.fixture
def make_batch_score_init_started_event(mock_run_context, make_configuration, make_metadata):
    setup_context_vars(make_configuration, make_metadata)

    return update_common_fields(BatchScoreInitStartedEvent())

@pytest.fixture
def make_batch_score_minibatch_completed_event(mock_run_context, make_configuration, make_metadata):
    configuration: Configuration = make_configuration
    setup_context_vars(configuration, make_metadata)

    event = BatchScoreMinibatchCompletedEvent(
        minibatch_id = '2',
        scoring_url = configuration.scoring_url,
        batch_pool = 'test_pool',
        quota_audience = 'test_audience',

        total_prompt_tokens = 50,
        total_completion_tokens = 1000,

        input_row_count=10,
        output_row_count=8,

        http_request_count=10,
        http_request_succeeded_count=5,
        http_request_user_error_count=3,
        http_request_system_error_count=2,
        http_request_retry_count=40,

        http_request_duration_p0_ms=0,
        http_request_duration_p50_ms=2,
        http_request_duration_p90_ms=5,
        http_request_duration_p95_ms=7,
        http_request_duration_p99_ms=10,
        http_request_duration_p100_ms=30,

        progress_duration_p0_ms=100,
        progress_duration_p50_ms=102,
        progress_duration_p90_ms=105,
        progress_duration_p95_ms=107,
        progress_duration_p99_ms=110,
        progress_duration_p100_ms=130,
    )

    return update_common_fields(event)

@pytest.fixture
def make_batch_score_minibatch_started_event(mock_run_context, make_configuration, make_metadata):
    configuration: Configuration = make_configuration
    setup_context_vars(configuration, make_metadata)

    event = BatchScoreMinibatchStartedEvent(
        minibatch_id = '2',
        scoring_url = configuration.scoring_url,
        batch_pool = 'test_pool',
        quota_audience = 'test_audience',
        input_row_count = 10)

    return update_common_fields(event)

def assert_run_context_fields(event: BatchScoreEvent):
    assert event.experiment_id == TEST_EXPERIMENT_ID
    assert event.parent_run_id == TEST_PARENT_RUN_ID
    assert event.resource_group == TEST_RESOURCE_GROUP
    assert event.run_id == TEST_RUN_ID
    assert event.subscription_id == TEST_SUBSCRIPTION_ID
    assert event.workspace_id == TEST_WORKSPACE_ID
    assert event.workspace_location == TEST_REGION
    assert event.workspace_name == TEST_WORKSPACE_NAME

def assert_common_fields(event: BatchScoreEvent):
    assert event.api_type == TEST_API_TYPE
    assert event.authentication_type == TEST_AUTHENTICATION_TYPE
    assert event.component_name == TEST_COMPONENT_NAME
    assert event.component_version == TEST_COMPONENT_VERSION
    assert event.endpoint_type == TEST_ENDPOINT_TYPE
    assert event.event_time == TEST_EVENT_TIME
    assert event.execution_mode == TEST_EXECUTION_MODE

def update_common_fields(event: BatchScoreEvent):
    event.event_time = TEST_EVENT_TIME
    event.execution_mode = TEST_EXECUTION_MODE

    return event
