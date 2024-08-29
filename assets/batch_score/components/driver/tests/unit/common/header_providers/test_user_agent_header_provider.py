# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for user agent header provider."""

import os
from unittest.mock import patch

import pytest

from src.batch_score.root.common.header_providers.user_agent_header_provider import (
    UserAgentHeaderProvider,
)

from tests.batch_score.fixtures.configuration import TEST_COMPONENT_VERSION

TEST_QUOTA_AUDIENCE = "test_audience"
TEST_RUN_ID = "test_run_id"
TEST_UA_SEGMENT = "test_ua_segment"
TEST_WORKSPACE_NAME = "test_ws"


@patch.dict(os.environ, {
    "AZUREML_RUN_ID": TEST_RUN_ID,
    "AZUREML_ARM_WORKSPACE_NAME": TEST_WORKSPACE_NAME
}, clear=True)
@pytest.mark.parametrize('expected_user_agent_string, user_agent_segment, quota_audience', [
     (f"BatchScore:{TEST_COMPONENT_VERSION}/{TEST_QUOTA_AUDIENCE}:{TEST_UA_SEGMENT}"
      f"/Run:{TEST_WORKSPACE_NAME}:{TEST_RUN_ID}",
      TEST_UA_SEGMENT,
      TEST_QUOTA_AUDIENCE),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/{TEST_QUOTA_AUDIENCE}"
      f"/Run:{TEST_WORKSPACE_NAME}:{TEST_RUN_ID}",
      None,
      TEST_QUOTA_AUDIENCE),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}/{TEST_UA_SEGMENT}", TEST_UA_SEGMENT, None),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}", None, None),
])
def test_get_headers(expected_user_agent_string, user_agent_segment, quota_audience):
    """Test get_headers method."""
    assert UserAgentHeaderProvider(
        component_version=TEST_COMPONENT_VERSION,
        user_agent_segment=user_agent_segment,
        quota_audience=quota_audience,
    ).get_headers() == {'User-Agent': expected_user_agent_string}
