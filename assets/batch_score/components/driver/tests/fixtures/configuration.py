# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock the component configuration."""

import pytest

from src.batch_score_oss.common.common_enums import ApiType, AuthenticationType
from src.batch_score_oss.common.configuration.configuration import Configuration
from src.batch_score_oss.common.configuration.metadata import Metadata

TEST_COMPONENT_NAME = "test_component_name"
TEST_COMPONENT_VERSION = "test_component_version"
TEST_SCORING_URI = "https://test.westus3.models.ai.azure.com/v1/chat/completions"


@pytest.fixture
def make_configuration():
    """Make a mock configuration object."""
    return Configuration(
        scoring_url=TEST_SCORING_URI,
        api_type=ApiType.ChatCompletion,
        authentication_type=AuthenticationType.ApiKey,
        async_mode=False,
    )


@pytest.fixture
def make_metadata():
    """Make a mock metadata object."""
    return Metadata(component_name=TEST_COMPONENT_NAME, component_version=TEST_COMPONENT_VERSION)
