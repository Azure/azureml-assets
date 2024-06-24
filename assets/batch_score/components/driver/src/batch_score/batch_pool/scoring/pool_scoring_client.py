# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Pool scoring client."""

from typing import Callable

import aiohttp

from ...common.scoring.scoring_client import ScoringClient
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import ScoringResult
from ...common.scoring.scoring_utils import RetriableType
from ...common.telemetry import logging_utils as lu
from ...mir.scoring.mir_scoring_client import MirScoringClient
from ..quota.quota_client import QuotaClient
from ..routing.routing_client import RoutingClient


class PoolScoringClient(ScoringClient):
    """Pool scoring client."""

    def __init__(
            self,
            create_mir_scoring_client: Callable[[str], MirScoringClient],
            quota_client: QuotaClient,
            routing_client: RoutingClient):
        """Initialize PoolScoringClient."""
        self._create_mir_scoring_client = create_mir_scoring_client
        self._routing_client = routing_client
        self._quota_client = quota_client
        self._scoring_client_per_endpoint: dict[str, MirScoringClient] = {}

    async def score(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            timeout: aiohttp.ClientTimeout,
            worker_id: str) -> ScoringResult:
        """Score the request."""
        scoring_url = await self._get_target_endpoint_url(session, scoring_request, worker_id)
        scoring_client = self._get_scoring_client(scoring_url)
        scoring_request.scoring_url = scoring_url
        quota_scope = await self._routing_client.get_quota_scope(session=session)

        async with self._quota_client.reserve_capacity(
                session=session,
                scope=quota_scope,
                request=scoring_request) as lease:
            try:
                self._routing_client.increment(scoring_request)
                result = await scoring_client.score(
                    session=session,
                    scoring_request=scoring_request,
                    timeout=timeout,
                    worker_id=worker_id)
                lease.report_result(result)
                return result
            finally:
                self._routing_client.decrement(scoring_request)

    def _get_scoring_client(self, scoring_url):
        if scoring_url not in self._scoring_client_per_endpoint:
            self._scoring_client_per_endpoint[scoring_url] = self._create_mir_scoring_client(scoring_url)

        return self._scoring_client_per_endpoint[scoring_url]

    async def _get_target_endpoint_url(self, session, scoring_request, worker_id):
        exclude_endpoint = None
        # If the most recent response_status for this request was insufficient, exclude the
        # corresponding endpoint on this next attempt
        if len(scoring_request.request_history) > 0:
            latest_attempt = scoring_request.request_history[-1]
            if latest_attempt.retriable_type == RetriableType.RETRY_ON_DIFFERENT_ENDPOINT:
                exclude_endpoint = latest_attempt.endpoint_base_url
                lu.get_logger().debug("{}: Excluding endpoint '{}' from consideration for the next attempt of "
                                      "this scoring_request".format(worker_id, exclude_endpoint))

        return await self._routing_client.get_target_endpoint(
            session=session,
            exclude_endpoint=exclude_endpoint)
