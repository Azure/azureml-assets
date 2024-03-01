# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for OpenAI header handler."""

import os
import pytest
from unittest import mock

from src.batch_score.header_handlers.open_ai.open_ai_header_handler import OpenAIHeaderHandler

from tests.fixtures.configuration import TEST_COMPONENT_VERSION

TEST_BATCH_POOL = "test_pool"
TEST_QUOTA_AUDIENCE = "test_audience"
TEST_RUN_ID = "test_run_id"
TEST_UA_SEGMENT = "test_ua_segment"
TEST_WORKSPACE_NAME = "test_ws"


@mock.patch.dict(os.environ, {
    "AZUREML_RUN_ID": TEST_RUN_ID,
    "AZUREML_ARM_WORKSPACE_NAME": TEST_WORKSPACE_NAME
}, clear=True)
@pytest.mark.parametrize('expected_user_agent_string, user_agent_segment, batch_pool, quota_audience', [
     (f"BatchScore:{TEST_COMPONENT_VERSION}/{TEST_BATCH_POOL}:{TEST_QUOTA_AUDIENCE}:{TEST_UA_SEGMENT}"
      f"/Run:{TEST_WORKSPACE_NAME}:{TEST_RUN_ID}",
      TEST_UA_SEGMENT,
      TEST_BATCH_POOL,
      TEST_QUOTA_AUDIENCE),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/{TEST_BATCH_POOL}:{TEST_QUOTA_AUDIENCE}"
      f"/Run:{TEST_WORKSPACE_NAME}:{TEST_RUN_ID}",
      None,
      TEST_BATCH_POOL,
      TEST_QUOTA_AUDIENCE),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}/{TEST_UA_SEGMENT}", TEST_UA_SEGMENT, None, None),
     (f"BatchScore:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}", None, None, None),
])
def test_get_user_agent(expected_user_agent_string, user_agent_segment, batch_pool, quota_audience):
    """Test get user agent."""
    # Arrange
    handler = OpenAIHeaderHandler(
        token_provider=None,
        component_version=TEST_COMPONENT_VERSION,
        user_agent_segment=user_agent_segment,
        batch_pool=batch_pool,
        quota_audience=quota_audience)

    user_agent = handler.get_user_agent()
    assert user_agent == expected_user_agent_string
