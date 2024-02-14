# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for quota client."""

import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientResponseError

from src.batch_score.batch_pool.quota.quota_client import (
    QuotaClient,
    QuotaUnavailableException,
)
from src.batch_score.common.scoring.scoring_request import ScoringRequest


@pytest.mark.asyncio
async def test_quota_client(make_quota_client, make_completion_header_handler):
    """Test quota client."""
    batch_pool = "cool-pool"

    client_session = FakeClientSession()
    token_provider = FakeTokenProvider()
    header_handler = make_completion_header_handler(token_provider=token_provider,
                                                    batch_pool=batch_pool,
                                                    quota_audience="cool-audience")
    scoring_request = ScoringRequest('{"prompt":"There was a farmer who had a cow"}')

    scope = f"endpointPools:{batch_pool}:trafficGroups:batch"

    quota_client: QuotaClient = make_quota_client(
        header_handler=header_handler,
        batch_pool=batch_pool,
        service_namespace="cool-namespace",
        quota_audience="cool-audience",
        quota_estimator="completion"
    )

    async with quota_client.reserve_capacity(client_session, scope, scoring_request):
        request_lease_url = ("https://azureml-drl-us.azureml.ms/ratelimiter/v1.0"
                             "/servicenamespaces/cool-namespace"
                             "/scopes/endpointPools:cool-pool:trafficGroups:batch"
                             "/audiences/cool-audience"
                             "/requestLease")
        assert ("POST", request_lease_url) in client_session.calls

    release_lease_url = ("https://azureml-drl-us.azureml.ms/ratelimiter/v1.0"
                         "/servicenamespaces/cool-namespace"
                         "/scopes/endpointPools:cool-pool:trafficGroups:batch"
                         "/audiences/cool-audience"
                         "/releaseLease")
    assert ("POST", release_lease_url) in client_session.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("error_headers, retry_after", [
    ({"Retry-After": 123}, 123),
    ({"x-ms-retry-after-ms": 750}, 0.75),
    ({"Retry-After": 123, "x-ms-retry-after-ms": 750}, 0.75),
])
async def test_quota_client_throttle(make_quota_client, make_completion_header_handler, error_headers, retry_after):
    """Test quota client throttle case."""
    batch_pool = "cool-pool"

    client_session = FakeClientSession(throttle_lease=True, error_headers=error_headers)
    token_provider = FakeTokenProvider()
    header_handler = make_completion_header_handler(token_provider=token_provider,
                                                    batch_pool=batch_pool,
                                                    quota_audience="cool-audience")
    scoring_request = ScoringRequest('{"prompt":"There was a farmer who had a cow"}')

    scope = f"endpointPools:{batch_pool}:trafficGroups:batch"

    quota_client: QuotaClient = make_quota_client(
        header_handler=header_handler,
        batch_pool=batch_pool,
        service_namespace="cool-namespace",
        quota_audience="cool-audience",
        quota_estimator="completion"
    )

    with pytest.raises(QuotaUnavailableException) as exc_info:
        async with quota_client.reserve_capacity(client_session, scope, scoring_request):
            pass

    assert exc_info.value.retry_after == retry_after

    # Completion estimator adds 10 to total tiktoken estimate
    assert scoring_request.estimated_cost == 8 + 10


@pytest.mark.asyncio
@pytest.mark.parametrize("input, expected_counts", [
    ("There was a farmer who had a cow", (8,)),
    (["There was a farmer who had a cow"], (8,)),
    (["There was a farmer who had a cow", "and bingo was her name oh.", "B", "I", "N", "G", "O"],
     (8, 7, 1, 1, 1, 1, 1)),
])
async def test_quota_client_embeddings(make_quota_client, make_completion_header_handler, input, expected_counts):
    """Test quota client embeddings case."""
    batch_pool = "cool-pool"

    client_session = FakeClientSession(throttle_lease=True, error_headers={"Retry-After": 123})
    token_provider = FakeTokenProvider()
    header_handler = make_completion_header_handler(token_provider=token_provider,
                                                    batch_pool=batch_pool,
                                                    quota_audience="cool-audience")
    inputstring = json.dumps({"input": input})
    scoring_request = ScoringRequest(inputstring)

    scope = f"endpointPools:{batch_pool}:trafficGroups:batch"

    quota_client: QuotaClient = make_quota_client(
        header_handler=header_handler,
        batch_pool=batch_pool,
        service_namespace="cool-namespace",
        quota_audience="cool-audience",
        quota_estimator="embeddings"
    )

    with pytest.raises(QuotaUnavailableException) as exc_info:
        async with quota_client.reserve_capacity(client_session, scope, scoring_request):
            pass

    assert exc_info.value.retry_after == 123

    # Embeddings have estimated costs of 1 and estimated tiktoken counts for the batch
    assert scoring_request.estimated_cost == 1
    assert scoring_request.estimated_token_counts == expected_counts


class FakeClientSession:
    """Mock client session."""

    def __init__(self, *, throttle_lease=False, error_headers=None):
        """Initialize FakeClientSession."""
        self._throttle_lease = throttle_lease
        self._error_headers = error_headers

        self.calls = []

    @asynccontextmanager
    async def post(self, url, *args, **kwargs):
        """Mock a POST request."""
        self.calls.append(("POST", url))
        if "requestLease" in url:
            if self._throttle_lease:
                error = ClientResponseError(None, None, status=429, headers=self._error_headers)
                yield FakeResponse(429, {}, error=error)
            else:
                yield FakeResponse(200, {"leaseId": "123", "leaseDuration": "99:99:99"})
        else:
            yield FakeResponse(404, {})


class FakeResponse:
    """Mock responsse."""

    def __init__(self, status, json, *, error=None):
        """Initialize FakeResponse."""
        self.status = status
        self._json = json
        self._error = error

    async def json(self):
        """Get json."""
        return self._json

    def raise_for_status(self):
        """Raise error."""
        if self._error is not None:
            raise self._error


class FakeTokenProvider:
    """Mock token provider."""

    def get_token(self, *args, **kwargs):
        """Mock get token."""
        pass
