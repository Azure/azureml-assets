# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for scoring client."""

import aiohttp
import pytest
from mock import MagicMock

from src.batch_score.batch_pool.scoring.scoring_client import ScoringClient
from src.batch_score.common.scoring.scoring_result import RetriableException


@pytest.mark.asyncio
async def test_score_once_pool_scenario(mock_get_logger, make_scoring_client, mock_get_quota_scope, mock__score_once):
    """Test score once pool scenario."""
    mock_client_session = MagicMock()
    mock_scoring_request = MagicMock()

    scoring_client: ScoringClient = make_scoring_client()

    scoring_result = await scoring_client.score_once(session=mock_client_session,
                                                     scoring_request=mock_scoring_request,
                                                     timeout=MagicMock())

    assert scoring_result


@pytest.mark.asyncio
async def test_score_once_endpoint_scenario(mock_get_logger, make_scoring_client, mock__score_once):
    """Test score once endpoint scenario."""
    mock_client_session = MagicMock()
    mock_scoring_request = MagicMock()
    mock_timeout = MagicMock()

    # Neither quota_client nor routing_client is provided
    scoring_client: ScoringClient = make_scoring_client(quota_client=None, routing_client=None)
    scoring_result = await scoring_client.score_once(session=mock_client_session,
                                                     scoring_request=mock_scoring_request,
                                                     timeout=mock_timeout)

    # quota_client is not provided
    scoring_client: ScoringClient = make_scoring_client(quota_client=None)
    scoring_result = await scoring_client.score_once(session=mock_client_session,
                                                     scoring_request=mock_scoring_request,
                                                     timeout=mock_timeout)

    # routing_client is not provided
    scoring_client: ScoringClient = make_scoring_client(routing_client=None)
    scoring_result = await scoring_client.score_once(session=mock_client_session,
                                                     scoring_request=mock_scoring_request,
                                                     timeout=mock_timeout)

    assert scoring_result

exceptions_to_raise = [
    aiohttp.ServerTimeoutError(),  # ServerTimeoutError extends TimeoutError
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
async def test_score_once_raises_retriable_exception(caplog,
                                                     make_completion_header_handler,
                                                     exception_to_raise,
                                                     mock_run_context):
    """Test score once raises retriable exception scenario."""
    def mock_post(**kwargs):
        """Mock post function."""
        raise exception_to_raise

    scoring_client = ScoringClient(header_handler=make_completion_header_handler(),
                                   quota_client=None,
                                   routing_client=None)
    aiohttp.ClientSession.post = MagicMock(side_effect=mock_post)

    async with aiohttp.ClientSession() as session:
        with pytest.raises(RetriableException):
            await scoring_client.score_once(session=session, scoring_request=MagicMock(), timeout=MagicMock())

    assert 'Score failed' in caplog.text
    assert type(exception_to_raise).__name__ in caplog.text


@pytest.mark.asyncio
async def test_score_once_raises_retriable_exception_until_max_retries(
    caplog,
    make_completion_header_handler,
    mock_run_context
):
    """Test score once raises retriable exception when retry count < max retries."""
    scoring_client = ScoringClient(
        header_handler=make_completion_header_handler(),
        quota_client=None,
        routing_client=None)
    session = MagicMock()
    session.post = MagicMock(side_effect=MockPost)
    scoring_request = MagicMock()
    scoring_request.retry_count = 1

    with pytest.raises(RetriableException):
        await scoring_client.score_once(session=session, scoring_request=scoring_request, timeout=MagicMock())

    assert 'Score failed' in caplog.text


@pytest.mark.asyncio
async def test_score_once_fails_when_max_retries_are_exhausted(
    caplog,
    make_completion_header_handler,
    mock_run_context
):
    """Test score once fails when retry count >= max retries."""
    scoring_client = ScoringClient(
        header_handler=make_completion_header_handler(),
        quota_client=None,
        routing_client=None)
    session = MagicMock()
    session.post = MagicMock(side_effect=MockPost)
    scoring_request = MagicMock()
    scoring_request.retry_count = 2

    await scoring_client.score_once(session=session, scoring_request=scoring_request, timeout=MagicMock())

    assert 'Score failed' in caplog.text


class MockClientResponse():
    """Mock client response."""

    def __init__(self, status, headers, reason):
        """Initialize mock client response."""
        self.status = status
        self.headers = headers
        self.reason = reason

    async def text(self):
        """Return mock client response text."""
        return 'Mock response text'


class MockPost():
    """Mock client session post."""

    def __init__(self, *args, **kwargs):
        """Initialize mock client session post."""
        pass

    async def __aenter__(self):
        """Return a mock client response when post is invoked."""
        return MockClientResponse(
            507,
            {'Content-Type': 'application/json', 'Random-Header': 'Random-Value'},
            'Insufficient Storage')

    async def __aexit__(self, exc_type, exc, tb):
        """Clean up any resources."""
        pass
