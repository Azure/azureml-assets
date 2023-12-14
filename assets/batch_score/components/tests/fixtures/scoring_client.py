# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock scoring client."""

import pytest

from src.batch_score.batch_pool.scoring.scoring_client import ScoringClient
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import ScoringResult


@pytest.fixture()
def make_scoring_client(make_completion_header_handler,
                        make_quota_client,
                        make_routing_client,
                        make_tally_failed_request_handler):
    """Mock scoring client."""
    def make(header_handler=make_completion_header_handler(),
             quota_client=make_quota_client(),
             routing_client=make_routing_client(),
             online_endpoint_url: str = None,
             tally_handler=make_tally_failed_request_handler()):
        """Make a mock scoring client."""
        client = ScoringClient(
            header_handler=header_handler,
            quota_client=quota_client,
            routing_client=routing_client,
            online_endpoint_url=online_endpoint_url,
            tally_handler=tally_handler,
        )
        return client
    return make


@pytest.fixture()
def mock__score_once(monkeypatch, make_scoring_result):
    """Mock score once."""
    state = {"raise_exception": None, "scoring_result": make_scoring_result()}

    async def __score_once(self, session, scoring_request: ScoringRequest, timeout, worker_id):
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

    monkeypatch.setattr("src.batch_score.batch_pool.scoring.scoring_client.ScoringClient._score_once", __score_once)
    return state
