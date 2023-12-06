# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import src.batch_score.main as main
from aiohttp import TraceConfig
from mock import MagicMock, patch

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
