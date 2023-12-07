# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.post_processing.callback_factory import CallbackFactory
from src.batch_score.common.post_processing.mini_batch_context import MiniBatchContext
from src.batch_score.common.scoring.scoring_result import ScoringResult
from tests.fixtures.input_transformer import FakeInputOutputModifier
from tests.fixtures.scoring_result import get_test_request_obj
from tests.fixtures.test_mini_batch_context import TestMiniBatchContext


def test_generate_callback_success(mock_get_logger, mock_get_events_client, make_input_transformer, make_scoring_result):
    mock_input_to_output_transformer = make_input_transformer(modifiers=[FakeInputOutputModifier()])

    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=1),
        target_result_len=2)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    gathered_result: list[ScoringResult] = [scoring_result.copy(), scoring_result.copy()]

    callback_factory = CallbackFactory(
        configuration=_get_test_configuration(),
        input_to_output_transformer=mock_input_to_output_transformer)

    callbacks = callback_factory.generate_callback()

    result_after_callback = callbacks(gathered_result, mini_batch_context)
    assert len(result_after_callback) == 2
    assert mock_get_events_client.emit_mini_batch_completed.called


def test_generate_callback_exception_with_mini_batch_id(mock_get_logger, mock_get_events_client, make_input_transformer, make_scoring_result):
    mock_input_to_output_transformer = make_input_transformer(modifiers=[FakeInputOutputModifier()])

    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=1),
        target_result_len=1)

    mini_batch_context.exception = __get_exception_with_traceback()

    gathered_result: list[ScoringResult] = [make_scoring_result(request_obj=get_test_request_obj())]

    callback_factory = CallbackFactory(
        configuration=_get_test_configuration(),
        input_to_output_transformer=mock_input_to_output_transformer)

    callbacks = callback_factory.generate_callback()

    result_after_callback = callbacks(gathered_result, mini_batch_context)
    assert len(result_after_callback) == 1

    args, kwargs = mock_get_events_client.emit_mini_batch_completed.call_args
    assert kwargs['exception'] == "ZeroDivisionError"
    assert kwargs['stacktrace'] is not None


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
