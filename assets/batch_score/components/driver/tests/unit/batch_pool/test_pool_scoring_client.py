# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for pool scoring client."""

import aiohttp
import pytest
from mock import AsyncMock, MagicMock

from src.batch_score.batch_pool.scoring.pool_scoring_client import PoolScoringClient
from src.batch_score.batch_pool.quota.quota_client import QuotaClient
from src.batch_score.batch_pool.routing.routing_client import RoutingClient
from src.batch_score.common.scoring.scoring_request import ScoringRequest
from src.batch_score.mir.scoring.mir_scoring_client import MirScoringClient


@pytest.mark.asyncio
async def test_score():
    """Test score."""
    # Arrange
    mir_scoring_client = AsyncMock(spec=MirScoringClient)
    mir_scoring_client.score.return_value = MagicMock()
    quota_client = AsyncMock(spec=QuotaClient)
    routing_client = AsyncMock(spec=RoutingClient)

    def create_mir_scoring_client(scoring_url):
        return mir_scoring_client

    scoring_client = PoolScoringClient(
        create_mir_scoring_client=create_mir_scoring_client,
        quota_client=quota_client,
        routing_client=routing_client)

    session = AsyncMock(spec=aiohttp.ClientSession)
    scoring_request = MagicMock(spec=ScoringRequest, request_history=[])
    score_kwargs = {
        "session": session,
        "scoring_request": scoring_request,
        "timeout": None,
        "worker_id": "1",
    }

    # Act
    result = await scoring_client.score(**score_kwargs)

    # Assert
    routing_client.get_quota_scope.assert_called_once_with(session=session)
    quota_client.reserve_capacity.assert_called_once_with(
        session=session,
        scope=routing_client.get_quota_scope.return_value,
        request=scoring_request)
    mir_scoring_client.score.assert_called_once_with(**score_kwargs)
    assert result == mir_scoring_client.score.return_value
