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
from src.batch_score.common.telemetry.events.batch_score_minibatch_started_event import (
    BatchScoreMinibatchStartedEvent,
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
    mock_os_environ_get.side_effect = environment_variables.get

    trace_configs = main.setup_trace_configs()

    assert trace_configs == None
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
    actual_result = main.get_return_value(dummy_return_data, output_behavior)
    if output_behavior == "summary_only":
        assert len(actual_result) == len(dummy_return_data)
        assert all(result_value == "True" for result_value in actual_result)
    else:
        assert actual_result == dummy_return_data

    assert mock_get_logger.info.called


def test_run_emit_minibatch_started_event(mock_run_context):
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
    # Arrange
    input_data, mini_batch_context = _setup_main()

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        main.enqueue(input_data=input_data, mini_batch_context=mini_batch_context)
    
    # Assert
    mock_generate_minibatch_summary.assert_not_called()


def test_enqueue_exception_generate_minibatch_summary(mock_run_context):
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


def _setup_main(par_exception=None):
    configuration = MagicMock()
    configuration.additional_properties = None
    configuration.batch_size_per_request = 1
    configuration.scoring_url = "https://scoring_url"
    configuration.batch_pool = "batch_pool"
    configuration.quota_audience = "quota_audience"
    main.configuration = configuration

    main.par = MagicMock()
    if par_exception is not None:
        main.par.enqueue.side_effect = par_exception

    input_data = pd.DataFrame()

    mini_batch_context = MagicMock()
    mini_batch_context.minibatch_index = 1

    return input_data, mini_batch_context