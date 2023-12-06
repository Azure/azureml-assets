# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import aiohttp
import pytest
from mock import MagicMock

from src.batch_score.batch_pool.scoring.scoring_client import ScoringClient
from src.batch_score.common.scoring.scoring_result import RetriableException


@pytest.mark.asyncio
async def test_score_once_pool_scenario(mock_get_logger, make_scoring_client, mock_get_quota_scope, mock__score_once):
    mock_client_session = MagicMock()
    mock_scoring_request = MagicMock()

    scoring_client: ScoringClient = make_scoring_client()

    scoring_result = await scoring_client.score_once(session=mock_client_session, scoring_request=mock_scoring_request, timeout=MagicMock())

    assert scoring_result

@pytest.mark.asyncio
async def test_score_once_endpoint_scenario(mock_get_logger, make_scoring_client, mock__score_once):
    mock_client_session = MagicMock()
    mock_scoring_request = MagicMock()
    mock_timeout = MagicMock()

    # Neither quota_client nor routing_client is provided
    scoring_client: ScoringClient = make_scoring_client(quota_client = None, routing_client = None)
    scoring_result = await scoring_client.score_once(session=mock_client_session, scoring_request=mock_scoring_request, timeout=mock_timeout)
    
    # quota_client is not provided
    scoring_client: ScoringClient = make_scoring_client(quota_client = None)
    scoring_result = await scoring_client.score_once(session=mock_client_session, scoring_request=mock_scoring_request, timeout=mock_timeout)
    
    # routing_client is not provided
    scoring_client: ScoringClient = make_scoring_client(routing_client = None)
    scoring_result = await scoring_client.score_once(session=mock_client_session, scoring_request=mock_scoring_request, timeout=mock_timeout)

    assert scoring_result

exceptions_to_raise = [
    aiohttp.ServerTimeoutError(), # ServerTimeoutError extends TimeoutError
    aiohttp.ServerConnectionError(),
    # See https://docs.aiohttp.org/en/stable/client_reference.html#hierarchy-of-exceptions 
    aiohttp.ClientOSError(), 
    aiohttp.ClientConnectorError(connection_key=None, os_error=OSError()),
    aiohttp.ClientConnectionError(),
    aiohttp.ClientPayloadError()
]
@pytest.mark.asyncio
@pytest.mark.parametrize(
    'caplog, make_completion_header_handler, exception_to_raise',
    [['caplog', 'make_completion_header_handler', e] for e in exceptions_to_raise],
    indirect=['caplog', 'make_completion_header_handler'],
)
async def test_score_once_raises_retriable_exception(caplog, make_completion_header_handler, exception_to_raise):
    def mock_post(**kwargs):
        raise exception_to_raise

    scoring_client = ScoringClient(header_handler=make_completion_header_handler(), quota_client=None, routing_client=None)
    aiohttp.ClientSession.post = MagicMock(side_effect=mock_post)

    async with aiohttp.ClientSession() as session:
        with pytest.raises(RetriableException):
            await scoring_client.score_once(session=session, scoring_request=MagicMock(), timeout=MagicMock())

    assert 'Score failed' in caplog.text
    assert type(exception_to_raise).__name__ in caplog.text

@pytest.mark.parametrize("time, expected_iters",
                         [(5, 1), (10, 1), (100, 4), (30*60*60, 9)])
def test_get_retry_timeout_generator(time, expected_iters):
    t = aiohttp.ClientTimeout(time)
    timeout_generator = ScoringClient.get_retry_timeout_generator(t)
    for i in range(expected_iters):
        timeout = next(timeout_generator)

    with pytest.raises(StopIteration):
        next(timeout_generator)

    assert timeout.total == time

@pytest.mark.parametrize("time, expected_iters",
                         [(5, 1), (10, 1), (100, 4), (30*60*60, 9)])
def test_get_next_retry_timeout(time, expected_iters):
    t = aiohttp.ClientTimeout(time)
    timeout_generator = ScoringClient.get_retry_timeout_generator(t)
    for i in range(expected_iters):
        timeout = ScoringClient.get_next_retry_timeout(timeout_generator)

    assert ScoringClient.get_next_retry_timeout(timeout_generator) is None

