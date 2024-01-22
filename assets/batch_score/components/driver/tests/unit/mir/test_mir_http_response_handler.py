# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for MIR HTTP response handler."""

import asyncio
import aiohttp
import pytest

from src.batch_score.common.scoring.http_scoring_response import HttpScoringResponse
from src.batch_score.mir.scoring.mir_http_response_handler import MirHttpResponseHandler
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import (
    RetriableException,
    ScoringResultStatus
)
from src.batch_score.common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from tests.unit.utils.scoring_result_utils import assert_scoring_result


end_time = 20.1
start_time = 14
scoring_url = 'https://mirendpoint@inference.com'


def test_handle_response_returns_success_result():
    # Arrange
    http_response = HttpScoringResponse(status=200, payload=['hello'], headers={'x-ms-client-request-id': '123'})
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    response_handler = MirHttpResponseHandler(TallyFailedRequestHandler(enabled=False))

    # Act
    scoring_result = response_handler.handle_response(http_response,
                                                      scoring_request,
                                                      start_time,
                                                      end_time,
                                                      scoring_url)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.SUCCESS,
        scoring_request,
        http_response,
        end_time,
        start_time
    )


def test_handle_response_retriable_failure_throws_exception():
    # Arrange
    http_response = HttpScoringResponse(status=403, payload=['hello'], headers={'x-ms-client-request-id': '123'})
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    response_handler = MirHttpResponseHandler(TallyFailedRequestHandler(enabled=False))

    # Act & Assert
    with pytest.raises(RetriableException) as ex:
        response_handler.handle_response(http_response, scoring_request, start_time, end_time, scoring_url)
    assert ex.value.status_code == 403


@pytest.mark.parametrize('enable_tally_handler', [True, False])
def test_handle_response_non_retriable_failure(enable_tally_handler):
    # Arrange
    http_response = HttpScoringResponse(status=500, payload=['hello'], headers={'x-ms-client-request-id': '123'})
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    response_handler = MirHttpResponseHandler(TallyFailedRequestHandler(enabled=enable_tally_handler))

    # Act
    scoring_result = response_handler.handle_response(http_response,
                                                      scoring_request,
                                                      start_time,
                                                      end_time,
                                                      scoring_url)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.FAILURE,
        scoring_request,
        http_response,
        end_time,
        start_time
    )

    assert scoring_result.omit == (True if enable_tally_handler else False)


@pytest.mark.parametrize('exception_to_throw', [
    (aiohttp.ClientConnectorError(connection_key=None, os_error=OSError())),
    (aiohttp.ClientPayloadError()),
    (aiohttp.ServerConnectionError()),
    (asyncio.TimeoutError())
])
def test_handler_retriable_exception_throws_exception(exception_to_throw):
    # Arrange
    http_response = HttpScoringResponse(exception=exception_to_throw, headers={'x-ms-client-request-id': '123'})
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    response_handler = MirHttpResponseHandler(TallyFailedRequestHandler(enabled=False))

    # Act & Assert
    with pytest.raises(RetriableException) as ex:
        response_handler.handle_response(http_response, scoring_request, start_time, end_time, scoring_url)
    assert ex.value.status_code == -408


@pytest.mark.parametrize('enable_tally_handler', [True, False])
def test_handler_non_retriable_exception_returns_failure(enable_tally_handler):
    # Arrange
    http_response = HttpScoringResponse(exception=Exception, headers={'x-ms-client-request-id': '123'})
    scoring_request = ScoringRequest(original_payload='{"prompt":"Test model"}')
    response_handler = MirHttpResponseHandler(TallyFailedRequestHandler(enabled=enable_tally_handler))

    # Act
    scoring_result = response_handler.handle_response(http_response,
                                                      scoring_request,
                                                      start_time,
                                                      end_time,
                                                      scoring_url)

    # Assert
    assert_scoring_result(
        scoring_result,
        ScoringResultStatus.FAILURE,
        scoring_request,
        http_response,
        end_time,
        start_time
    )

    assert scoring_result.omit == (True if enable_tally_handler else False)
