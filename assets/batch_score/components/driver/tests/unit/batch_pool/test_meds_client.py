# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for MEDS client."""

import aiohttp
import pytest
from unittest.mock import patch, MagicMock

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.batch_pool.meds_client import MEDSClient

from tests.fixtures.client_response import FakeResponse


@pytest.mark.asyncio
class TestMEDSClient:
    """Test MEDS client."""

    @pytest.mark.parametrize(
        "quota_audience, response_status, response_body, expected_connection_string",
        [
            # Connection string is missing.
            (
                "quota_audience",
                200,
                {
                    'clientSettings': {}
                },
                None,
            ),
            (
                "quota_audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/another_quota_audience': "ANOTHER-CONNECTION-STRING"
                    }
                },
                None,
            ),
            # Connection string is found.
            (
                "quota_audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING"
                    }
                },
                "MOCK-CONNECTION-STRING",
            ),
            (
                "quota_audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING",
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/another_quota_audience': "ANOTHER-CONNECTION-STRING"
                    }
                },
                "MOCK-CONNECTION-STRING",
            ),
            # Lookup is case insensitive.
            (
                "QUOTA_AUDIENCE",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING"
                    }
                },
                "MOCK-CONNECTION-STRING",
            ),
            (
                "Quota_Audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING"
                    }
                },
                "MOCK-CONNECTION-STRING",
            ),
            (
                "Quota_Audience",
                200,
                {
                    'clientSettings': {
                        'Application_Insights_Connection_String/QUOTA_AUDIENCE': "MOCK-CONNECTION-STRING"
                    }
                },
                "MOCK-CONNECTION-STRING",
            ),
            # But space, underscore, and hyphen are not interchangeable.
            (
                "quota audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING"
                    }
                },
                None,
            ),
            (
                "quota-audience",
                200,
                {
                    'clientSettings': {
                        'APPLICATION_INSIGHTS_CONNECTION_STRING/quota_audience': "MOCK-CONNECTION-STRING"
                    }
                },
                None,
            ),
            # IF MEDS responds with non-200, return None.
            ("quota_audience", 401, {}, None),
            ("quota_audience", 403, {}, None),
            ("quota_audience", 404, {}, None),
            ("quota_audience", 500, {}, None),
        ],
    )
    async def test_get_application_insights_connection_string(
            self,
            quota_audience,
            response_status,
            response_body,
            expected_connection_string):
        """Test get application insights connection string."""
        # Arrange
        meds_client = MEDSClient(
            header_provider=MagicMock(),
            configuration=self._get_pool_configuration(quota_audience))

        async with aiohttp.ClientSession() as session:
            with patch.object(session, "post") as mock_post:
                mock_post.return_value.__aenter__.return_value = FakeResponse(
                    status=response_status,
                    json=response_body)

                # Act
                connection_string = await meds_client.get_application_insights_connection_string(session)

        # Assert
        assert connection_string == expected_connection_string

    def _get_pool_configuration(self, quota_audience):
        return Configuration(
            service_namespace="service_namespace",
            batch_pool="batch_pool",
            quota_audience=quota_audience,
        )
