import pytest

from src.batch_score.common.common_enums import ApiType, AuthenticationType
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.configuration.metadata import Metadata

TEST_COMPONENT_NAME = "test_component_name"
TEST_COMPONENT_VERSION = "test_component_version"

@pytest.fixture
def make_configuration():
    return Configuration(
        scoring_url = "https://sunjoli-aoai.openai.azure.com/openai/deployments/turbo/chat/completions?api-version=2023-03-15-preview",
        api_type = ApiType.ChatCompletion,
        authentication_type = AuthenticationType.ApiKey,
        async_mode=False,
    )

@pytest.fixture
def make_metadata():
    return Metadata(component_name=TEST_COMPONENT_NAME, component_version=TEST_COMPONENT_VERSION)
