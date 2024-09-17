# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for callback factory."""

import pytest

from unittest.mock import patch, MagicMock

from src.batch_score_oss.root.common.configuration.configuration import Configuration
from src.batch_score_oss.root.common.post_processing.callback_factory import CallbackFactory
from src.batch_score_oss.root.common.post_processing.mini_batch_context import MiniBatchContext
from src.batch_score_oss.root.common.scoring.scoring_result import ScoringResult
from src.batch_score_oss.root.common.telemetry.events import event_utils
from src.batch_score_oss.root.common.post_processing.output_handler import (
    SingleFileOutputHandler,
    SeparateFileOutputHandler
)
from tests.batch_score.fixtures.input_transformer import FakeInputOutputModifier
from tests.batch_score.fixtures.scoring_result import get_test_request_obj
from tests.batch_score.fixtures.test_mini_batch_context import TestMiniBatchContext


def test_generate_callback_success(mock_get_logger,
                                   mock_get_events_client,
                                   make_input_transformer,
                                   make_scoring_result):
    """Test generate callback success case."""
    # Arrange
    mock_input_to_output_transformer = make_input_transformer(modifiers=[FakeInputOutputModifier()])

    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=1),
        target_result_len=2)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    gathered_result: list[ScoringResult] = [scoring_result.copy(), scoring_result.copy()]

    callback_factory = CallbackFactory(
        configuration=_get_test_configuration(),
        output_handler=SingleFileOutputHandler(),
        input_to_output_transformer=mock_input_to_output_transformer)

    callbacks = callback_factory.generate_callback()

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        result_after_callback = callbacks(gathered_result, mini_batch_context)

    # Assert
    assert len(result_after_callback) == 2
    assert mock_get_events_client.emit_mini_batch_completed.called
    mock_generate_minibatch_summary.assert_called_once_with(
        minibatch_id=1,
        output_row_count=2,
    )


def test_generate_callback_exception_with_mini_batch_id(mock_get_logger,
                                                        mock_get_events_client,
                                                        make_input_transformer,
                                                        make_scoring_result):
    """Test generate callback exception case."""
    # Arrange
    mock_input_to_output_transformer = make_input_transformer(modifiers=[FakeInputOutputModifier()])

    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=1),
        target_result_len=1)

    mini_batch_context.exception = __get_exception_with_traceback()

    gathered_result: list[ScoringResult] = [make_scoring_result(request_obj=get_test_request_obj())]

    callback_factory = CallbackFactory(
        configuration=_get_test_configuration(),
        output_handler=SingleFileOutputHandler(),
        input_to_output_transformer=mock_input_to_output_transformer)

    callbacks = callback_factory.generate_callback()

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        result_after_callback = callbacks(gathered_result, mini_batch_context)

    # Assert
    assert len(result_after_callback) == 1

    args, kwargs = mock_get_events_client.emit_mini_batch_completed.call_args
    assert kwargs['exception'] == "ZeroDivisionError"
    assert kwargs['stacktrace'] is not None
    mock_generate_minibatch_summary.assert_called_once_with(
        minibatch_id=1,
        output_row_count=1,
    )


@pytest.mark.parametrize("split_output, use_single_file_output_handler, use_separate_file_output_handler", [
    (False, True, False),
    (True, False, True)
])
def test_output_handler(
        split_output,
        use_single_file_output_handler,
        use_separate_file_output_handler,
        make_input_transformer,
        make_scoring_result):
    """Test output handler."""
    mock_input_to_output_transformer = make_input_transformer(modifiers=[FakeInputOutputModifier()])

    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=1),
        target_result_len=2)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    gathered_result: list[ScoringResult] = [scoring_result.copy(), scoring_result.copy()]

    with patch(
        "tests.batch_score.unit.common.post_processing.test_callback_factory.SeparateFileOutputHandler",
        return_value=MagicMock()
      ) as mock_separate_file_output_handler, \
        patch(
            "tests.batch_score.unit.common.post_processing.test_callback_factory.SingleFileOutputHandler",
            return_value=MagicMock()
          ) as mock_single_file_output_handler:

        test_configuration = _get_test_configuration_for_output_handler(split_output)
        if test_configuration.split_output:
            output_handler = SeparateFileOutputHandler()
        else:
            output_handler = SingleFileOutputHandler()

        callback_factory = CallbackFactory(
            configuration=test_configuration,
            output_handler=output_handler,
            input_to_output_transformer=mock_input_to_output_transformer)

        callbacks = callback_factory.generate_callback()

        _ = callbacks(gathered_result, mini_batch_context)

        assert mock_separate_file_output_handler.called == use_separate_file_output_handler
        assert mock_single_file_output_handler.called == use_single_file_output_handler

        assert mock_separate_file_output_handler.return_value.save_mini_batch_results.called == \
            use_separate_file_output_handler
        assert mock_single_file_output_handler.return_value.save_mini_batch_results.called == \
            use_single_file_output_handler


def __get_exception_with_traceback():
    try:
        1/0
    except Exception as e:
        return e


def _get_test_configuration() -> Configuration:
    return Configuration(
        batch_size_per_request=1,
        mini_batch_results_out_directory="test_mini_batch_results_out_directory",
        output_behavior="summary_only",
        save_mini_batch_results="disabled",
    )


def _get_test_configuration_for_output_handler(split_output: bool) -> Configuration:
    return Configuration(
        batch_size_per_request=1,
        mini_batch_results_out_directory="test_mini_batch_results_out_directory",
        output_behavior="summary_only",
        split_output=split_output,
        save_mini_batch_results="enabled",
    )
