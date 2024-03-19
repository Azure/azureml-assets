# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MEDS client."""

import os
from aiohttp import ClientSession

from ..common.configuration.configuration import Configuration
from ..common.header_providers.header_provider import HeaderProvider
from ..common.telemetry import logging_utils as lu


class MEDSClient:
    """Client for Model Endpoint Discovery Service (MEDS)."""

    def __init__(self, header_provider: HeaderProvider, configuration: Configuration):
        """Initialize MEDSClient."""
        self._header_provider = header_provider
        self._quota_audience = configuration.quota_audience

        base_url = os.environ.get(
            "BATCH_SCORE_ROUTING_BASE_URL",
            "https://model-endpoint-discovery.api.azureml.ms/modelendpointdiscovery")

        self._url = (
            f"{base_url}/v1.0"
            f"/serviceNamespaces/{configuration.service_namespace}"
            f"/endpointPools/{configuration.batch_pool}"
            "/listEndpoints")

    async def get_application_insights_connection_string(self, session: ClientSession) -> "str | None":
        """Get a client setting from MEDS."""
        client_settings = await self._get_client_settings(session)

        # Allow case-insensitive key lookup by converting all dictionary keys to lowercase.
        client_settings = {k.lower(): v for k, v in client_settings.items()}

        # And converting the key we are looking for to lowercase.
        key = f"APPLICATION_INSIGHTS_CONNECTION_STRING/{self._quota_audience}".lower()

        connection_string = client_settings.get(key)

        if connection_string:
            lu.get_logger().info("Application Insights connection string found in MEDS.")
        else:
            lu.get_logger().warning(f"Application Insights connection string not found in MEDS for key '{key}'.")

        return connection_string

    async def _get_client_settings(self, session: ClientSession) -> 'dict[str, str]':
        """Get client settings from MEDS."""
        headers = self._header_provider.get_headers()
        async with session.post(url=self._url, headers=headers, json={"trafficGroup": 'batch'}) as response:
            if response.status != 200:
                response_body = await response.text()
                lu.get_logger().warning(f"Failed to get client settings from MEDS: "
                                        f"Response status: {response.status} "
                                        f"Response body: {response_body}")
                client_settings = {}
            else:
                response_body = await response.json()
                client_settings = response_body.get('clientSettings', {})

        return client_settings
