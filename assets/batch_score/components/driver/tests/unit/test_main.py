# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for main."""

import pytest

import pandas as pd
from aiohttp import TraceConfig
from mock import MagicMock, patch

from tests.fixtures.geneva_event_listener import mock_import
with patch('importlib.import_module', side_effect=mock_import):
    import src.batch_score.main as main

from src.batch_score.batch_pool.meds_client import MEDSClient
from src.batch_score.common.common_enums import EndpointType
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.post_processing.output_handler import SeparateFileOutputHandler, SingleFileOutputHandler
from src.batch_score.common.telemetry.events import event_utils
from src.batch_score.common.telemetry.trace_configs import (
    ConnectionCreateEndTrace,
    ConnectionCreateStartTrace,
    ConnectionReuseconnTrace,
    ExceptionTrace,
    RequestEndTrace,
    RequestRedirectTrace,
    ResponseChunkReceivedTrace,
)

DISABLED_TRACE_CONFIG_TESTS = [
    {"BATCH_SCORE_TRACE_LOGGING": "fAlSe"},  # Environment variable set to case-insensitive false bool
    {"BATCH_SCORE_TRACE_LOGGING": "fooBar123"},  # Environment variable set to jibberish
    {},  # Environment variable not set
]


@pytest.mark.parametrize("environment_variables", DISABLED_TRACE_CONFIG_TESTS)
@patch("os.environ.get")
def test_setup_trace_configs_disabled_scenario(mock_os_environ_get: MagicMock, mock_get_logger,
                                               environment_variables: "dict[str, str]"):
    """Test setup trace configs disabled scenario."""
    mock_os_environ_get.side_effect = environment_variables.get

    trace_configs = main.setup_trace_configs()

    assert trace_configs is None
    assert mock_get_logger.info.called


ENABLED_TRACE_CONFIG_TESTS = [
    [{"BATCH_SCORE_TRACE_LOGGING": "TrUe"},
     [ExceptionTrace, ResponseChunkReceivedTrace, RequestEndTrace, RequestRedirectTrace, ConnectionCreateStartTrace,
      ConnectionCreateEndTrace, ConnectionReuseconnTrace]],  # Environment variable set to case-insensitive true bool
]


@pytest.mark.parametrize("environment_variables, expected_trace_configs", ENABLED_TRACE_CONFIG_TESTS)
@patch("os.environ.get")
def test_setup_trace_configs_enabled_scenario(mock_os_environ_get: MagicMock, mock_get_logger,
                                              environment_variables: "dict[str, str]",
                                              expected_trace_configs: "list[TraceConfig]"):
    """Test setup trace configs enabled scenario."""
    mock_os_environ_get.side_effect = environment_variables.get

    trace_configs = main.setup_trace_configs()

    for trace_config in trace_configs:
        assert type(trace_config) in expected_trace_configs
    assert len(trace_configs) == len(expected_trace_configs)
    assert mock_get_logger.info.called


GET_RETURN_VALUE_TEST_CASES = [
    [["some fun string", "another fun string", "the last fun string"], "append_row"],
    [["some fun string", "another fun string", "the last fun string"], "summary_only"],
    # The pipeline validation should prevent any other output_behavior values,
    # but we use the default behavior just in case
    [["some fun string", "another fun string", "the last fun string"], "any_other_value"],
    [[], "append_row"],
    [[], "summary_only"]
]


@pytest.mark.parametrize("dummy_return_data, output_behavior", GET_RETURN_VALUE_TEST_CASES)
def test_get_return_value(mock_get_logger, dummy_return_data: "list[str]", output_behavior: str):
    """Test get return value."""
    actual_result = main.get_return_value(dummy_return_data, output_behavior)
    if output_behavior == "summary_only":
        assert len(actual_result) == len(dummy_return_data)
        assert all(result_value == "True" for result_value in actual_result)
    else:
        assert actual_result == dummy_return_data

    assert mock_get_logger.info.called


@pytest.mark.parametrize("split_output, use_single_file_output_handler, use_separate_file_output_handler", [
    (False, True, False),
    (True, False, True)
])
def test_output_handler_interface(
        split_output: bool,
        use_single_file_output_handler: bool,
        use_separate_file_output_handler: bool):
    """Test output handler interface."""
    with patch(
        "tests.unit.test_main.SeparateFileOutputHandler"
      ) as mock_separate_file_output_handler, \
        patch(
        "tests.unit.test_main.SingleFileOutputHandler"
      ) as mock_single_file_output_handler:

        input_data, mini_batch_context = _setup_main()

        main.configuration.split_output = split_output
        main.configuration.save_mini_batch_results = "enabled"
        main.configuration.mini_batch_results_out_directory = "driver/tests/unit/unit_test_results/"
        if main.configuration.split_output:
            main.output_handler = SeparateFileOutputHandler(main.configuration.batch_size_per_request,
                                                            main.configuration.input_schema_version)
        else:
            main.output_handler = SingleFileOutputHandler(main.configuration.batch_size_per_request,
                                                          main.configuration.input_schema_version)
        main.run(input_data=input_data, mini_batch_context=mini_batch_context)

        assert mock_separate_file_output_handler.called == use_separate_file_output_handler
        assert mock_single_file_output_handler.called == use_single_file_output_handler

        assert mock_separate_file_output_handler.return_value.save_mini_batch_results.called == \
            use_separate_file_output_handler
        assert mock_single_file_output_handler.return_value.save_mini_batch_results.called == \
            use_single_file_output_handler


def test_run_emit_minibatch_started_event(mock_run_context):
    """Test emit minibatch started event."""
    # Arrange
    input_data, mini_batch_context = _setup_main()

    # Act
    with patch.object(event_utils, 'emit_event') as mock_emit_event:
        main.run(input_data=input_data, mini_batch_context=mini_batch_context)

    # Assert
    assert mock_emit_event.call_count == 1
    assert 'batch_score_event' in mock_emit_event.call_args_list[0].kwargs
    event = mock_emit_event.call_args_list[0].kwargs['batch_score_event']
    assert event.batch_pool == main.configuration.batch_pool
    assert event.input_row_count == 0
    assert event.minibatch_id == 1
    assert event.quota_audience == main.configuration.quota_audience
    assert event.scoring_url == main.configuration.scoring_url


def test_run_generate_minibatch_summary(mock_run_context):
    """Test run generate minibatch summary."""
    # Arrange
    input_data, mini_batch_context = _setup_main()

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        main.run(input_data=input_data, mini_batch_context=mini_batch_context)

    # Assert
    mock_generate_minibatch_summary.assert_called_once_with(
        minibatch_id=1,
        output_row_count=0,
    )


def test_enqueue_emit_minibatch_started_event(mock_run_context):
    """Test enqueue emit minibatch started event."""
    # Arrange
    input_data, mini_batch_context = _setup_main()

    # Act
    with patch.object(event_utils, 'emit_event') as mock_emit_event:
        main.enqueue(input_data=input_data, mini_batch_context=mini_batch_context)

    # Assert
    assert mock_emit_event.call_count == 1
    assert 'batch_score_event' in mock_emit_event.call_args_list[0].kwargs
    event = mock_emit_event.call_args_list[0].kwargs['batch_score_event']
    assert event.batch_pool == main.configuration.batch_pool
    assert event.input_row_count == 0
    assert event.minibatch_id == 1
    assert event.quota_audience == main.configuration.quota_audience
    assert event.scoring_url == main.configuration.scoring_url


def test_enqueue_no_exception_does_not_generate_minibatch_summary(mock_run_context):
    """Test enqueue no exception does not generate minibatch summary."""
    # Arrange
    input_data, mini_batch_context = _setup_main()

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        main.enqueue(input_data=input_data, mini_batch_context=mini_batch_context)

    # Assert
    mock_generate_minibatch_summary.assert_not_called()


def test_enqueue_exception_generate_minibatch_summary(mock_run_context):
    """Test enqueue exception generate minibatch summary."""
    # Arrange
    input_data, mini_batch_context = _setup_main(par_exception=Exception)

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        with pytest.raises(Exception):
            main.enqueue(input_data=input_data, mini_batch_context=mini_batch_context)

    # Assert
    mock_generate_minibatch_summary.assert_called_once_with(
        minibatch_id=1,
        output_row_count=0,
    )


# Dummy Application Insights connection strings
conn1 = ("InstrumentationKey=11111111-1111-1111-1111-111111111111;"
         "IngestionEndpoint=https://centralus-1.in.applicationinsights.azure.com/;"
         "LiveEndpoint=https://centralus.livediagnostics.monitor.azure.com/")
conn2 = ("InstrumentationKey=22222222-2222-2222-2222-222222222222;"
         "IngestionEndpoint=https://centralus-2.in.applicationinsights.azure.com/;"
         "LiveEndpoint=https://centralus.livediagnostics.monitor.azure.com/")


@pytest.mark.asyncio
@pytest.mark.parametrize("from_configuration, endpoint_type, from_MEDS, expected", [
    # When a connection string is present in the configuration, it is used.
    (conn1, EndpointType.AOAI, None, conn1),
    (conn1, EndpointType.BatchPool, None, conn1),
    # When missing and the endpoint type is BatchPool, the value from MEDS is used.
    (None, EndpointType.BatchPool, None, None),    # Missing in MEDS
    (None, EndpointType.BatchPool, conn2, conn2),  # Present in MEDS
    # When missing and the endpoint type is not BatchPool, None is returned without using MEDS.
    (None, EndpointType.AOAI, None, None),
    (None, EndpointType.AOAI, conn2, None),
])
async def test_get_application_insights_connection_string(
        from_configuration,
        endpoint_type,
        from_MEDS,
        expected):
    """Test get application insights connection string."""
    # Arrange
    configuration = Configuration(
        app_insights_connection_string=from_configuration,
        batch_pool="batch_pool" if endpoint_type == EndpointType.BatchPool else None,
        quota_audience="quota_audience" if endpoint_type == EndpointType.BatchPool else None,
        service_namespace="service_namespace" if endpoint_type == EndpointType.BatchPool else None)
    with patch.object(MEDSClient, "get_application_insights_connection_string") as mock_get_client_setting:
        mock_get_client_setting.return_value = from_MEDS

        # Act
        connection_string = await main.get_application_insights_connection_string(
            configuration=configuration,
            metadata=MagicMock(),
            token_provider=MagicMock())

    # Assert
    assert connection_string == expected


def _setup_main(par_exception=None):
    configuration = MagicMock()
    configuration.additional_properties = None
    configuration.batch_size_per_request = 1
    configuration.scoring_url = "https://scoring_url"
    configuration.batch_pool = "batch_pool"
    configuration.quota_audience = "quota_audience"
    configuration.input_schema_version = 1
    main.configuration = configuration

    main.input_handler = MagicMock()

    main.par = MagicMock()
    if par_exception is not None:
        main.par.enqueue.side_effect = par_exception

    input_data = pd.DataFrame()

    mini_batch_context = MagicMock()
    mini_batch_context.minibatch_index = 1

    return input_data, mini_batch_context
