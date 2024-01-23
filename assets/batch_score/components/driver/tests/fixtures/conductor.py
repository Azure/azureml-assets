# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock conductor."""

import asyncio
import json

import pytest

from src.batch_score.common import constants
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.parallel.conductor import Conductor
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus,
)


@pytest.fixture
def make_conductor(make_routing_client, make_scoring_client):
    """Make a mock conductor."""
    async_mode_outer = False
    conductor = None

    def make(loop=asyncio.get_event_loop(),
             routing_client=None,
             scoring_client=make_scoring_client(),
             segment_large_requests="disabled",
             segment_max_token_size=None,
             initial_worker_count=1,
             max_worker_count=10,
             trace_configs=None,
             max_retry_time_interval=None,
             async_mode=False):
        configuration = Configuration(
            async_mode=async_mode,
            initial_worker_count=initial_worker_count,
            max_retry_time_interval=max_retry_time_interval,
            max_worker_count=max_worker_count,
            request_path=constants.DV_COMPLETION_API_PATH,
            segment_large_requests=segment_large_requests,
            segment_max_token_size=segment_max_token_size,
        )

        nonlocal async_mode_outer
        async_mode_outer = async_mode

        nonlocal conductor
        conductor = Conductor(
            configuration=configuration,
            loop=loop,
            routing_client=routing_client or make_routing_client(),
            scoring_client=scoring_client,
            trace_configs=trace_configs,
        )

        return conductor

    yield make

    # When the conductor is run in async mode, it starts an infinite loop.
    # Stop it before exiting the test. If we don't, the test suite will hang.
    if conductor and async_mode_outer:
        conductor._Conductor__loop.stop()


@pytest.fixture
def mock_run(monkeypatch):
    """Mock run function."""
    passed_requests = []

    async def _run(self, requests: "list[ScoringRequest]") -> "list[ScoringResult]":
        passed_requests.extend(requests)
        return [ScoringResult(status=ScoringResultStatus.SUCCESS,
                              response_body={"usage": {}},
                              omit=False,
                              start=0,
                              end=0,
                              request_obj=json.loads(scoring_request.cleaned_payload),
                              request_metadata=scoring_request.request_metadata,
                              response_headers=None,
                              num_retries=0) for scoring_request in requests]

    monkeypatch.setattr("src.batch_score.common.parallel.conductor.Conductor.run", _run)
    return passed_requests
