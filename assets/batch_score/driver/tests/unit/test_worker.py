# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from collections import deque

import pandas as pd
import pytest
from mock import patch

from src.batch_score.batch_pool.quota import QuotaUnavailableException
from src.batch_score.common.parallel.request_metrics import RequestMetrics
from src.batch_score.common.parallel.worker import QueueItem, Worker
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import (
    RetriableException,
    ScoringResultStatus,
)


@pytest.mark.asyncio
async def test_successful_scoring_appends_result_no_segmentation(
    mock_get_events_client, 
    make_worker, 
    mock_get_logger,
    mock__score_once):
    
    metrics = await _run_worker(make_worker)

    assert not metrics.empty
    assert metrics.iloc[0]['response_code'] == ScoringResultStatus.SUCCESS


@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_successful_scoring_appends_result_with_segmentation(
        monkeypatch,
        mock_get_events_client,
        make_worker,
        mock_get_logger,
        make_scoring_result):

    async def mock_score(*args, **kwargs):
        return make_scoring_result(request_obj={"prompt": "payload"})

    monkeypatch.setattr("src.batch_score.common.scoring.segmented_score_context.SegmentedScoreContext.score_next_once", mock_score)
    with patch('src.batch_score.common.scoring.segmented_score_context.SegmentedScoreContext.has_more',
            side_effect=[True, True, True, False]) as has_more:
        metrics = await _run_worker(make_worker, segment_large_requests='enabled')

    assert not metrics.empty
    assert metrics.iloc[0]['response_code'] == ScoringResultStatus.SUCCESS


@pytest.mark.asyncio
async def test_request_exceeds_max_retry_time_interval_and_fails(
    mock_get_events_client, 
    make_worker, 
    mock_get_logger):

    # 1 second maximum
    max_retry_time_interval = 1

    payloads = ['{"fake": "payload"}']

    queue_item = QueueItem(
        scoring_request=ScoringRequest(original_payload=payloads[0]))

    # update duration to exceed timeout value of 1 second
    queue_item.scoring_request.scoring_duration = 1.1

    queue = deque()
    queue.append(queue_item)
    
    worker = make_worker(
        scoring_request_queue=queue,
        max_retry_time_interval=max_retry_time_interval)

    ret = await worker.start()

    assert mock_get_logger.error.called
    assert mock_get_events_client.emit_row_completed.called

    assert (worker._Worker__request_metrics._RequestMetrics__df["response_code"].head(1) == ScoringResultStatus.FAILURE).all()

@pytest.mark.asyncio
async def test_model_429_does_not_contribute_to_request_total_wait_time(
    mock_get_events_client,
    make_worker,
    mock_get_logger,
    mock__score_once,
    mock_get_client_setting):

    mock__score_once['raise_exception'] = RetriableException(status_code=424, model_response_code='429', retry_after=0.01)
    mock_get_client_setting['COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME'] = 'true'

    metrics = await _run_worker(make_worker)

    assert not metrics.empty
    assert metrics.iloc[0]['request_total_wait_time'] == 0

@pytest.mark.asyncio
async def test_quota_429_contributes_to_request_total_wait_time(
    mock_get_events_client,
    make_worker,
    mock_get_logger,
    mock__score_once,
    mock_get_client_setting):

    mock__score_once['raise_exception'] = QuotaUnavailableException(retry_after=0.01)
    mock_get_client_setting['COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME'] = 'true'

    metrics = await _run_worker(make_worker)

    assert not metrics.empty
    assert metrics.iloc[0]['request_total_wait_time'] > 0

fixture_names = [
    'mock_get_events_client',
    'make_worker',
    'mock_get_logger',
    'mock__score_once',
    'mock_get_client_setting',
    'monkeypatch',
]
no_deployments_test_cases = [
    # Both env vars absent. Defaults to no wait.
    (
        *fixture_names,
        {}, # env vars
        0,  # expected wait time
    ),
    # Poll env var absent. Back off env var present. Defaults to no wait.
    (
        *fixture_names,
        {
            'BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF': '123',
        },
        0,
    ),
    # Poll env var present. Back off env var absent. Defaults to worker.NO_DEPLOYMENTS_BACK_OFF.
    (
        *fixture_names,
        {
            'BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS': 'true',
        },
        1,
    ),
    # Both env vars present. Uses provided back off as wait time.
    (
        *fixture_names,
        {
            'BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS': 'true',
            'BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF': '2',
        },
        2,
    ),
]
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mock_get_events_client, make_worker, mock_get_logger, mock__score_once, mock_get_client_setting, monkeypatch, env_vars, expected_wait_time",
    no_deployments_test_cases,
    indirect=['mock_get_events_client', 'make_worker', 'mock_get_logger', 'mock__score_once', 'mock_get_client_setting', 'monkeypatch'],
)
async def test_no_deployments_in_traffic_group(
    mock_get_events_client,
    make_worker,
    mock_get_logger,
    mock__score_once,
    mock_get_client_setting,
    monkeypatch,
    env_vars,
    expected_wait_time):

    for var, value in env_vars.items():
        monkeypatch.setenv(var, value)
    
    if 'BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF' not in env_vars:
        monkeypatch.setattr(Worker, 'NO_DEPLOYMENTS_BACK_OFF', expected_wait_time)

    mock__score_once['raise_exception'] = RetriableException(status_code=404, response_payload='Specified traffic group could not be found')

    metrics = await _run_worker(make_worker)

    assert not metrics.empty
    assert metrics.iloc[0]['request_total_wait_time'] == expected_wait_time

async def _run_worker(make_worker, segment_large_requests='disabled'):
    payload = '{"fake": "payload"}'
    queue_item = QueueItem(scoring_request=ScoringRequest(payload))
    queue = deque([queue_item])

    start_time = pd.Timestamp.utcnow()
    request_metrics = RequestMetrics()
    assert request_metrics.get_metrics(start_time).empty
    
    worker = make_worker(
        scoring_request_queue=queue,
        segment_large_requests=segment_large_requests,
        max_retry_time_interval=0.001, # A small timeout keeps the test fast.
        request_metrics=request_metrics)
    
    worker._Worker__count_only_quota_429s_toward_total_request_time = True

    # Disable quota client so we don't make quota requests during the unit test.
    worker._Worker__scoring_client._ScoringClient__quota_client = None

    _ = await worker.start()

    metrics = request_metrics.get_metrics(start_time)
    return metrics
