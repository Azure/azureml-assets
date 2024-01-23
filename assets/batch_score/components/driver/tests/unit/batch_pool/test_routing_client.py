# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for routing client."""

import aiohttp
import pytest

from src.batch_score.batch_pool.routing.routing_client import (
    InvalidPoolRoutes,
    RoutingClient,
)
from src.batch_score.common.configuration.client_settings import ClientSettingsKey


@pytest.mark.asyncio
class TestRoutingClient:
    """Test routing client."""

    @pytest.mark.parametrize(
        "client_settings_from_endpoint_discovery_service, key, expected_value",
        [
            (
                {},
                ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME,
                None,
            ),
            (
                {
                    'unrecognized key': 'unrevealed value',
                },
                ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME,
                None,
            ),
            (
                {
                    ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME: 'true',
                },
                ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME,
                'true',
            ),
            (
                {
                    ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME: 'true',
                    'unrecognized key': 'unrevealed value',
                },
                ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME,
                'true',
            ),
            (
                {
                    ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME: 'true',
                    ClientSettingsKey.CONGESTION_THRESHOLD_P90_WAIT_TIME: '100',
                },
                ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME,
                'true',
            ),
        ],
    )
    async def test_get_client_settings(self, client_settings_from_endpoint_discovery_service, key, expected_value):
        """Test get client settings."""
        routing_client = RoutingClient(
            service_namespace="MOCK-NAMESPACE",
            target_batch_pool="MOCK-POOL",
            header_handler=None,
            request_path="MOCK-PATH"
        )

        routing_client._RoutingClient__client_settings = client_settings_from_endpoint_discovery_service

        assert expected_value == routing_client.get_client_setting(key)

    # If the exception handling fails, this test will hang indefinitely.
    # So we expect the test to pass in 5 seconds if the code is correct.
    @pytest.mark.timeout(5)
    async def test_exception_handling_for__check_and_refresh_pool_routes_no_exception_raised(
            self,
            mock_refresh_pool_routes):
        """Test check and refresh pool routes exception handling."""
        routing_client = RoutingClient(
            service_namespace="MOCK-NAMESPACE",
            target_batch_pool="MOCK-POOL",
            header_handler=None,
            request_path="MOCK-PATH"
        )

        mock_refresh_pool_routes['exception_to_raise'] = None

        await routing_client._RoutingClient__check_and_refresh_pool_routes(session=None)

    # If the exception handling fails, this test will hang indefinitely.
    # So we expect the test to pass in 5 seconds if the code is correct.
    @pytest.mark.timeout(5)
    async def test_exception_handling_for__check_and_refresh_pool_routes_exception_is_raised(
            self,
            mock_refresh_pool_routes):
        """Test check and refresh pool routes exception raised."""
        routing_client = RoutingClient(
            service_namespace="MOCK-NAMESPACE",
            target_batch_pool="MOCK-POOL",
            header_handler=None,
            request_path="MOCK-PATH"
        )

        mock_refresh_pool_routes['exception_to_raise'] = Exception("MOCK-EXCEPTION")

        await routing_client._RoutingClient__check_and_refresh_pool_routes(session=None)

    # If the exception handling fails, this test will hang indefinitely.
    # So we expect the test to pass in 40 seconds if the code is correct.
    @pytest.mark.timeout(5)
    async def test_get_quota_scope_InvalidPoolRoutes_raised(
            self,
            monkeypatch,
            make_completion_header_handler):
        """Test get quota scope invalid pool routes raised."""
        routing_client = RoutingClient(
            service_namespace="MOCK-NAMESPACE",
            target_batch_pool="MOCK-POOL",
            header_handler=make_completion_header_handler(),
            request_path="MOCK-PATH"
        )
        routing_client._RoutingClient__MAX_RETRY = 2
        routing_client._RoutingClient__RETRY_INTERVAL = 1

        def mock_invalid_pool_routes(*args, **kwargs):
            raise InvalidPoolRoutes('Failed to get routes')

        monkeypatch.setattr("aiohttp.ClientSession.post", mock_invalid_pool_routes)

        with pytest.raises(InvalidPoolRoutes) as excinfo:
            async with aiohttp.ClientSession() as session:
                await routing_client.get_quota_scope(session=session)
        assert "Valid Pool Routes could not be obtained" in str(excinfo.value)
