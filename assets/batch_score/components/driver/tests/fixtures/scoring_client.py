# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock scoring client."""

import pytest
from src.batch_score.root.common.header_providers.user_agent_header_provider import UserAgentHeaderProvider
from src.batch_score.root.common.scoring.http_response_handler import HttpResponseHandler
from src.batch_score.root.common.scoring.generic_scoring_client import GenericScoringClient
from src.batch_score.root.common.scoring.scoring_request import ScoringRequest
from src.batch_score.root.common.scoring.scoring_result import ScoringResult


@pytest.fixture()
def make_generic_scoring_client():
    """Mock scoring client."""
    def make():
        return GenericScoringClient(
            header_provider=UserAgentHeaderProvider(component_version="1.0"),
            http_response_handler=HttpResponseHandler(),
            scoring_url="null.inference.io"
        )
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

    monkeypatch.setattr("src.batch_score.root.common.scoring.generic_scoring_client.GenericScoringClient.score", score)
    return state
