# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock worker."""

from collections import deque

import aiohttp
import pytest

from src.batch_score_oss.common import constants
from src.batch_score_oss.common.configuration.client_settings import NullClientSettingsProvider
from src.batch_score_oss.common.configuration.configuration import Configuration
from src.batch_score_oss.common.parallel.request_metrics import RequestMetrics
from src.batch_score_oss.common.parallel.worker import Worker


@pytest.fixture
def make_worker(make_generic_scoring_client):
    """Mock worker."""
    def make(
            scoring_client=None,
            client_session=None,
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
            request_path=constants.OSS_COMPLETIONS_API_PATH,
            segment_large_requests=segment_large_requests,
            segment_max_token_size=segment_max_token_size,
        )
        return Worker(
            configuration=configuration,
            scoring_client=scoring_client or make_generic_scoring_client(),
            client_session=client_session or aiohttp.ClientSession(),
            client_settings_provider=client_settings_provider or NullClientSettingsProvider(),
            scoring_request_queue=scoring_request_queue,
            scoring_result_queue=scoring_result_queue,
            request_metrics=request_metrics,
            id=id,
        )

    return make


@pytest.fixture
def mock_get_client_setting(monkeypatch):
    """Mock get client setting."""
    state = {}

    def _get_client_setting(self, key):
        return state.get(key)

    monkeypatch.setattr(
        "src.batch_score_oss.common.configuration.client_settings.NullClientSettingsProvider.get_client_setting",
        _get_client_setting)

    return state
