from collections import deque

import aiohttp
import pytest

from src.batch_score.batch_pool.scoring.scoring_client import ScoringClient
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.parallel.request_metrics import RequestMetrics
from src.batch_score.common.parallel.worker import Worker


@pytest.fixture
def make_worker(make_scoring_client, make_routing_client):
    def make(
        scoring_client=None,
        client_session=aiohttp.ClientSession(),
        client_settings_provider=None,
        scoring_request_queue=deque(),
        scoring_result_queue=deque(),
        request_metrics=RequestMetrics(),
        segment_large_requests="disabled",
        segment_max_token_size=None,
        id=1,
        max_retry_time_interval=1):

        configuration = Configuration(
            async_mode=False,
            max_retry_time_interval=max_retry_time_interval,
            request_path=ScoringClient.DV_COMPLETION_API_PATH,
            segment_large_requests=segment_large_requests,
            segment_max_token_size=segment_max_token_size,
        )
        return Worker(
            configuration=configuration,
            scoring_client=scoring_client or make_scoring_client(),
            client_session=client_session,
            client_settings_provider=client_settings_provider or make_routing_client(),
            scoring_request_queue=scoring_request_queue, 
            scoring_result_queue=scoring_result_queue,
            request_metrics=request_metrics,
            id=id,
        )
    
    return make