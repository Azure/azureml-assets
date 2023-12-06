# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from collections import deque

import pytest
from src.batch_score.common.post_processing.gatherer import Gatherer
from src.batch_score.common.post_processing.mini_batch_context import MiniBatchContext
from src.batch_score.common.scoring.scoring_result import ScoringResultStatus
from tests.fixtures.scoring_result import get_test_request_obj
from tests.fixtures.test_mini_batch_context import TestMiniBatchContext


def __get_callback(ret, mini_batch_context):
    return ret

def __get_callback_throwing_exception():
    raise Exception("test")

def test_add_empty_result(mock_get_logger):
    mini_batch_id = 1
    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=mini_batch_id),
        target_result_len=2)

    gatherer = Gatherer(scoring_result_queue=deque(), failed_scoring_result_queue=deque(), callback=__get_callback)
    gatherer.add_empty_result(mini_batch_context)

    assert mini_batch_id in gatherer._Gatherer__finished_minibatch_id_set
    assert gatherer._Gatherer__metrics.handled_minibatch_count == 1
    assert gatherer.get_returned_minibatch_count() == 0
    assert len(gatherer.get_finished_minibatch_result()) == 1

# Since the gather runs forever, we need to stop the gatherer manually.
async def wait_and_stop(gatherer):
    await asyncio.sleep(1)
    gatherer._Gatherer__working = False

async def run_once(gatherer):
    await asyncio.gather(
        wait_and_stop(gatherer),
        asyncio.create_task(gatherer.run()))

@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_run_success(mock_get_logger, make_scoring_result):
    mini_batch_id = 1
    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=mini_batch_id),
        target_result_len=1)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    scoring_result.mini_batch_context = mini_batch_context

    scoring_result_queue = deque()
    scoring_result_queue.append(scoring_result)

    gatherer = Gatherer(scoring_result_queue=scoring_result_queue, failed_scoring_result_queue=deque(), callback=__get_callback)
    await run_once(gatherer)

    assert gatherer.get_returned_minibatch_count() == 0

    finished_minibatch_result = gatherer.get_finished_minibatch_result()
    assert len(finished_minibatch_result) == 1
    assert len(finished_minibatch_result[mini_batch_id]["ret"]) == 1

    assert gatherer.get_returned_minibatch_count() == 1

    finished_minibatch_result = gatherer.get_finished_minibatch_result()
    assert len(finished_minibatch_result) == 0
    assert gatherer.get_returned_minibatch_count() == 1

@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_run_failed_results(mock_get_logger, make_scoring_result):
    mini_batch_id = 1
    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=mini_batch_id),
        target_result_len=2)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    scoring_result.mini_batch_context = mini_batch_context

    scoring_result_queue = deque()
    scoring_result_queue.append(scoring_result)

    failed_scoring_result = make_scoring_result(status=ScoringResultStatus.FAILURE, request_obj=get_test_request_obj())
    failed_scoring_result.mini_batch_context = mini_batch_context

    failed_scoring_result_queue = deque()
    failed_scoring_result_queue.append(failed_scoring_result)

    gatherer = Gatherer(scoring_result_queue, failed_scoring_result_queue, __get_callback)
    await run_once(gatherer)

    assert gatherer.get_returned_minibatch_count() == 0

    finished_minibatch_result = gatherer.get_finished_minibatch_result()
    assert len(finished_minibatch_result) == 1
    assert len(finished_minibatch_result[mini_batch_id]["ret"]) == 2

    assert gatherer.get_returned_minibatch_count() == 1

@pytest.mark.timeout(5)
@pytest.mark.asyncio
async def test_run_exception(mock_get_logger, make_scoring_result):
    mini_batch_id = 1
    mini_batch_context = MiniBatchContext(
        raw_mini_batch_context=TestMiniBatchContext(minibatch_index=mini_batch_id),
        target_result_len=1)

    scoring_result = make_scoring_result(request_obj=get_test_request_obj())
    scoring_result.mini_batch_context = mini_batch_context

    scoring_result_queue = deque()
    scoring_result_queue.append(scoring_result)

    gatherer = Gatherer(scoring_result_queue=scoring_result_queue, failed_scoring_result_queue=deque(), callback=__get_callback_throwing_exception)
    await run_once(gatherer)

    assert gatherer.get_returned_minibatch_count() == 0

    finished_minibatch_result = gatherer.get_finished_minibatch_result()
    assert len(finished_minibatch_result) == 1
    assert finished_minibatch_result[mini_batch_id]['exception'] is not None