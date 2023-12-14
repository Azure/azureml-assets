# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock quota client."""

import pytest

from src.batch_score.batch_pool.quota.quota_client import QuotaClient


@pytest.fixture
def make_quota_client(make_completion_header_handler):
    """Mock quota client."""
    def make(header_handler=make_completion_header_handler(),
             service_namespace: str = None,
             quota_audience: str = None,
             batch_pool: str = None,
             quota_estimator: str = "completion") -> QuotaClient:
        """Make a mock quota client."""
        quota_client = QuotaClient(
            header_handler=header_handler,
            service_namespace=service_namespace,
            quota_audience=quota_audience,
            batch_pool=batch_pool,
            quota_estimator=quota_estimator
        )

        return quota_client

    return make
