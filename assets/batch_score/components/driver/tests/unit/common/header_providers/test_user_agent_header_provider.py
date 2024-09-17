# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for user agent header provider."""

import os
from unittest.mock import patch

import pytest

from src.batch_score_oss.root.common import constants
from src.batch_score_oss.root.common.header_providers.user_agent_header_provider import (
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
@pytest.mark.parametrize('expected_user_agent_string, user_agent_segment', [
     (f"{constants.BATCH_SCORE_USER_AGENT}:{TEST_COMPONENT_VERSION}"
      f"/Run:{TEST_RUN_ID}/{TEST_UA_SEGMENT}",
      TEST_UA_SEGMENT),
     (f"{constants.BATCH_SCORE_USER_AGENT}:{TEST_COMPONENT_VERSION}"
      f"/Run:{TEST_RUN_ID}",
      None),
     (f"{constants.BATCH_SCORE_USER_AGENT}:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}/{TEST_UA_SEGMENT}", TEST_UA_SEGMENT),
     (f"{constants.BATCH_SCORE_USER_AGENT}:{TEST_COMPONENT_VERSION}/Run:{TEST_RUN_ID}", None),
])
def test_get_headers(expected_user_agent_string, user_agent_segment):
    """Test get_headers method."""
    assert UserAgentHeaderProvider(
        component_version=TEST_COMPONENT_VERSION,
        user_agent_segment=user_agent_segment,
    ).get_headers() == {'User-Agent': expected_user_agent_string}
