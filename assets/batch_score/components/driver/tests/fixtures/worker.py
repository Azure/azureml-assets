# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock worker."""

from collections import deque
from unittest.mock import MagicMock

import aiohttp
import pytest

from src.batch_score.common import constants
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.parallel.request_metrics import RequestMetrics
from src.batch_score.common.parallel.worker import Worker


@pytest.fixture
def make_worker(make_pool_scoring_client, make_routing_client):
    """Mock worker."""
    def make(
            scoring_client=None,
            client_session=None,
            client_settings_provider=None,
            scoring_request_queue=None,
            scoring_result_queue=None,
            request_metrics=None,
            segment_large_requests="disabled",
            segment_max_token_size=None,
            id=1,
            max_retry_time_interval=1):

        configuration = Configuration(
            async_mode=False,
            max_retry_time_interval=max_retry_time_interval,
            request_path=constants.DV_COMPLETION_API_PATH,
            segment_large_requests=segment_large_requests,
            segment_max_token_size=segment_max_token_size,
        )
        return Worker(
            configuration=configuration,
            scoring_client=scoring_client or make_pool_scoring_client(),
            client_session=client_session or MagicMock(spec=aiohttp.ClientSession),
            client_settings_provider=client_settings_provider or make_routing_client(),
            scoring_request_queue=scoring_request_queue if scoring_request_queue is not None else deque(),
            scoring_result_queue=scoring_result_queue if scoring_result_queue is not None else deque(),
            request_metrics=request_metrics or RequestMetrics(),
            id=id,
        )

    return make
