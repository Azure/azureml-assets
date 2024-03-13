# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock scoring client."""

from unittest.mock import MagicMock

import pytest

from src.batch_score.batch_pool.scoring.pool_scoring_client import PoolScoringClient
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import ScoringResult


@pytest.fixture()
def make_pool_scoring_client(make_quota_client, make_routing_client):
    """Mock scoring client."""
    def make(quota_client=None, routing_client=None):
        return PoolScoringClient(
            create_mir_scoring_client=lambda _: MagicMock(),
            quota_client=quota_client or make_quota_client(),
            routing_client=routing_client or make_routing_client())
    return make


@pytest.fixture()
def mock_score(monkeypatch, make_scoring_result):
    """Mock score once."""
    state = {"raise_exception": None, "scoring_result": make_scoring_result()}

    async def score(self, session, scoring_request: ScoringRequest, timeout, worker_id):
        if state["raise_exception"]:
            # This delay ensures that the scoring duration (end - start) isn't zero.
            # Without this delay, the scoring duration of the request will never increase.
            # So the request will never time out and the while loop in worker.py will run for infinity. :)
            import time
            time.sleep(0.1)
            raise state["raise_exception"]

        given_scoring_result: ScoringResult = state["scoring_result"]
        given_scoring_result.request_obj = scoring_request.original_payload_obj
        given_scoring_result.request_metadata = scoring_request.request_metadata

        return given_scoring_result

    monkeypatch.setattr("src.batch_score.batch_pool.scoring.pool_scoring_client.PoolScoringClient.score", score)
    return state
