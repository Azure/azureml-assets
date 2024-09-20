# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for conductor."""

import asyncio
import os

import pytest
from mock import MagicMock, patch

from src.batch_score.common.parallel.conductor import Conductor
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import ScoringResult
from src.batch_score.common.telemetry.events import event_utils


@pytest.mark.asyncio
async def test_run_returns_same_number_of_results_as_requests(
    make_conductor,
    mock_get_logger,
    mock_get_events_client,
    monkeypatch,
):
    """Test run returns same number of results as requests."""
    conductor: Conductor = make_conductor(
        loop=asyncio.get_event_loop(),
        routing_client=None,
        scoring_client=None,
        segment_large_requests="disabled",
        segment_max_token_size=None,
        initial_worker_count=1,
        max_worker_count=10,
        trace_configs=None
    )

    # Mock the worker's start method to prevent it from actually running.
    # Return a dummy result instead.
    async def mock_worker_start(self):
        while len(self._Worker__scoring_request_queue) > 0:
            queue_item = self._Worker__scoring_request_queue.pop()
            self._Worker__scoring_result_queue.append(ScoringResult.Failed(queue_item.scoring_request))

    monkeypatch.setattr('src.batch_score.common.parallel.worker.Worker.start', mock_worker_start)

    # Minibatch has one request.
    # There should be one result.
    requests1: list[ScoringRequest] = [
        ScoringRequest(original_payload='{"fake": "payload"}'),
    ]

    results1 = await conductor.run(requests=requests1)
    assert len(results1) == len(requests1)

    # Run the conductor a second time, this time with 2 requests in the minibatch.
    # We're reusing the conductor and its worker.
    # There should be 2 results.
    requests2: list[ScoringRequest] = 2 * [
        ScoringRequest(original_payload='{"fake": "payload"}'),
    ]

    results2 = await conductor.run(requests=requests2)
    assert len(results2) == len(requests2)

    # Run with 3 requests in a minibatch.
    # There should be 3 results.
    requests3: list[ScoringRequest] = 3 * [
        ScoringRequest(original_payload='{"fake": "payload"}'),
    ]

    results3 = await conductor.run(requests=requests3)
    assert len(results3) == len(requests3)


def test_enqueue_empty_minibatch_generate_minibatch_summary(
    make_conductor,
    mock_get_logger,
    mock_get_events_client,
    monkeypatch,
):
    """Test enqueue empty minibatch generate minibatch summary."""
    # Arrange
    conductor: Conductor = make_conductor(async_mode=True)

    mini_batch_context = MagicMock()
    mini_batch_context.minibatch_index = 1

    # Act
    with patch.object(event_utils, 'generate_minibatch_summary') as mock_generate_minibatch_summary:
        conductor.enqueue(
            requests=[],
            failed_results=[],
            mini_batch_context=mini_batch_context,
        )

    # Assert
    mock_generate_minibatch_summary.assert_called_once_with(
        minibatch_id=1,
        output_row_count=0,
    )


@pytest.mark.parametrize("segment_large_requests, max_retry_time_interval",
                         [("disabled", None), ("disabled", 102), ("enabled", None), ("enabled", 102)])
def test_get_session_timeout_env_var(monkeypatch, make_conductor, segment_large_requests, max_retry_time_interval):
    """Test get session timeout env var case."""
    conductor: Conductor = make_conductor(
        segment_large_requests=segment_large_requests,
        segment_max_token_size=600,
        initial_worker_count=1,
        max_retry_time_interval=max_retry_time_interval,
    )
    # Test env var takes precedence even when max retry or segmentation are set
    monkeypatch.setattr(os, 'environ', {'BATCH_SCORE_REQUEST_TIMEOUT': '101'})

    session = conductor._Conductor__get_session()
    assert session.timeout.total == 101


@pytest.mark.parametrize("segment_large_requests", ["disabled", "enabled"])
def test_get_session_timeout_max_retry(make_conductor, segment_large_requests):
    """Test get session timeout max retry case."""
    conductor: Conductor = make_conductor(
        segment_large_requests=segment_large_requests,
        segment_max_token_size=600,
        initial_worker_count=1,
        max_retry_time_interval=102,
    )
    # Test max retry takes precedence even when segmentation enabled
    session = conductor._Conductor__get_session()
    assert session.timeout.total == 102


@pytest.mark.parametrize("segment_large_requests, expected_default", [("disabled", 1800), ("enabled", 600)])
def test_get_session_timeout_defaults(make_conductor, segment_large_requests, expected_default):
    """Test get session timeout default case."""
    conductor: Conductor = make_conductor(
        segment_large_requests=segment_large_requests,
        segment_max_token_size=600,
        initial_worker_count=1
    )
    session = conductor._Conductor__get_session()
    assert session.timeout.total == expected_default
