# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for aoai response handler."""

import asyncio
import aiohttp

import pytest

from mock import ANY, patch

from src.batch_score.aoai.scoring.aoai_response_handler import AoaiHttpResponseHandler
from src.batch_score.common.scoring.http_scoring_response import HttpScoringResponse
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from src.batch_score.common.scoring.scoring_result import (
    RetriableException,
    ScoringResultStatus
)
from src.batch_score.common.telemetry.events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent
from tests.unit.utils.scoring_result_utils import assert_scoring_result


test_end_time = 20.1
test_start_time = 14
test_x_ms_client_request_id = "test_client_id"
test_worker_id = 2
test_scoring_url = "test_scoring_url"
test_prompt_tokens = 10
test_completion_tokens = 15
test_segment_count = 5

exceptions_to_raise = [
    aiohttp.ServerConnectionError(),
    aiohttp.ClientConnectorError(connection_key=None, os_error=OSError()),
    aiohttp.ClientConnectionError(),
    asyncio.TimeoutError
]


def test_handle_response_returns_success_result(mock_run_context):
    """Test handle response returns success result."""
    # Arrange
    response_handler = AoaiHttpResponseHandler(TallyFailedRequestHandler(enabled=False))
    http_response = HttpScoringResponse(
        status=200,
        payload={"usage": {
            "prompt_tokens": test_prompt_tokens,
            "completion_tokens": test_completion_tokens,
            "total_tokens": test_prompt_tokens + test_completion_tokens}})
    scoring_request = _get_test_scoring_request()
    expected_request_completed_event = _get_expected_request_completed_event(response_handler,
                                                                             scoring_request,
                                                                             http_response)

    # Act
    with patch("src.batch_score.common.telemetry.events.event_utils.emit_event") as mock_emit_event:
        scoring_result = response_handler.handle_response(
            http_response,
            scoring_request,
            test_x_ms_client_request_id,
            test_start_time,
            test_end_time,
            test_worker_id)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.SUCCESS,
        scoring_request,
        http_response,
        test_end_time,
        test_start_time
    )
    mock_emit_event.assert_called_with(batch_score_event=expected_request_completed_event)


def test_handle_response_non_retriable_exception_returns_failure(mock_run_context):
    """Test handle response non retriable exception returns failure."""
    # Arrange
    http_response = HttpScoringResponse(exception=Exception)
    scoring_request = _get_test_scoring_request()
    response_handler = AoaiHttpResponseHandler(TallyFailedRequestHandler(enabled=False))
    expected_request_completed_event = _get_expected_request_completed_event(response_handler,
                                                                             scoring_request,
                                                                             http_response)

    # Act
    with patch("src.batch_score.common.telemetry.events.event_utils.emit_event") as mock_emit_event:
        scoring_result = response_handler.handle_response(
            http_response,
            scoring_request,
            test_x_ms_client_request_id,
            test_start_time,
            test_end_time,
            test_worker_id)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.FAILURE,
        scoring_request,
        http_response,
        test_end_time,
        test_start_time
    )
    mock_emit_event.assert_called_with(batch_score_event=expected_request_completed_event)


@pytest.mark.parametrize('exception_to_throw', [
    (aiohttp.ClientConnectorError(connection_key=None, os_error=OSError())),
    (aiohttp.ServerConnectionError()),
    (asyncio.TimeoutError())
])
def test_handle_response_retriable_exception_throws_exception(exception_to_throw: Exception, mock_run_context):
    """Test handle response retriable exception throws exception."""
    # Arrange
    http_response = HttpScoringResponse(exception=exception_to_throw, status=500)
    scoring_request = _get_test_scoring_request()
    response_handler = AoaiHttpResponseHandler(TallyFailedRequestHandler(enabled=False))
    expected_request_completed_event = _get_expected_request_completed_event(response_handler,
                                                                             scoring_request,
                                                                             http_response)

    # Act & Assert
    with patch("src.batch_score.common.telemetry.events.event_utils.emit_event") as mock_emit_event:
        with pytest.raises(RetriableException) as ex:
            response_handler.handle_response(
                http_response,
                scoring_request,
                test_x_ms_client_request_id,
                test_start_time,
                test_end_time,
                test_worker_id)
    assert ex.value.status_code == 500
    mock_emit_event.assert_called_with(batch_score_event=expected_request_completed_event)


# This is an actual failed response payload from a model.
# We use the 'r' prefix to create a raw string so we don't have to escape the escape characters in the string.
failed_response_payload = r'{\n  "error": {\n    "message": "This model\'s maximum context length is 16385 tokens, ' \
                          r'however you requested 9000027 tokens (27 in your prompt; 9000000 for the completion). ' \
                          r'Please reduce your prompt; or completion length.",\n    "type": "invalid_request_error"' \
                          r',\n    "param": null,\n    "code": null\n  }\n}\n'


@pytest.mark.parametrize('status_code', [(408), (429), (500), (502), (503), (504)])
def test_handle_response_retriable_status_code_throws_exception(status_code, mock_run_context):
    """Test handle response retriable status code throws exception."""
    # Arrange
    http_response = HttpScoringResponse(status=status_code, payload=failed_response_payload)
    scoring_request = _get_test_scoring_request()
    response_handler = AoaiHttpResponseHandler(TallyFailedRequestHandler(enabled=False))
    expected_request_completed_event = _get_expected_request_completed_event(response_handler,
                                                                             scoring_request,
                                                                             http_response)

    # Act & Assert
    with patch("src.batch_score.common.telemetry.events.event_utils.emit_event") as mock_emit_event:
        with pytest.raises(RetriableException) as ex:
            response_handler.handle_response(
                http_response,
                scoring_request,
                test_x_ms_client_request_id,
                test_start_time,
                test_end_time,
                test_worker_id)
    assert ex.value.status_code == status_code
    mock_emit_event.assert_called_with(batch_score_event=expected_request_completed_event)


@pytest.mark.parametrize('status_code', [400, 401, 403, 404, 409])
@pytest.mark.parametrize('tally_handler_enable', [True, False])
def test_handle_response_non_retriable_status_code_returns_failure(status_code,
                                                                   tally_handler_enable,
                                                                   mock_run_context):
    """Test handle response non retriable status code returns failure."""
    # Arrange
    http_response = HttpScoringResponse(status=status_code, payload=failed_response_payload)
    scoring_request = _get_test_scoring_request()
    response_handler = AoaiHttpResponseHandler(TallyFailedRequestHandler(enabled=tally_handler_enable))
    expected_request_completed_event = _get_expected_request_completed_event(response_handler,
                                                                             scoring_request,
                                                                             http_response)

    # Act
    with patch("src.batch_score.common.telemetry.events.event_utils.emit_event") as mock_emit_event:
        scoring_result = response_handler.handle_response(
            http_response,
            scoring_request,
            test_x_ms_client_request_id,
            test_start_time,
            test_end_time,
            test_worker_id)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.FAILURE,
        scoring_request,
        http_response,
        test_end_time,
        test_start_time
    )
    assert scoring_result.omit == tally_handler_enable
    mock_emit_event.assert_called_with(batch_score_event=expected_request_completed_event)


def _get_expected_request_completed_event(
    aoai_response_handler: AoaiHttpResponseHandler,
    scoring_request: ScoringRequest,
    http_response: HttpScoringResponse
):
    if http_response.headers:
        model_response_code = http_response.headers.get("ms-azureml-model-error-statuscode", None)
    else:
        model_response_code = None

    if http_response.payload is not None and "usage" in http_response.payload:
        prompt_tokens = test_prompt_tokens
        completion_tokens = test_completion_tokens
    else:
        prompt_tokens = None
        completion_tokens = None

    return BatchScoreRequestCompletedEvent(
        event_time=ANY,
        minibatch_id=ANY,
        input_row_id=scoring_request.internal_id,
        x_ms_client_request_id=test_x_ms_client_request_id,
        worker_id=test_worker_id,
        scoring_url=test_scoring_url,
        is_successful=True if http_response.status == 200 else False,
        is_retriable=aoai_response_handler.is_retriable(http_response.status),
        response_code=http_response.status,
        model_response_code=model_response_code,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        duration_ms=(test_end_time - test_start_time) * 1000,
        segmented_request_id=test_segment_count
    )


def _get_test_scoring_request() -> ScoringRequest:
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    scoring_request.scoring_url = test_scoring_url
    scoring_request.segment_id = test_segment_count

    return scoring_request
