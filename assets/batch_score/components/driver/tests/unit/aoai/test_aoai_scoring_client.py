# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for aoai scoring client."""

import pytest

from src.batch_score.aoai.scoring.aoai_scoring_client import AoaiScoringClient
from src.batch_score.common.configuration.configuration_parser import ConfigurationParser
from src.batch_score.common.scoring.tally_failed_request_handler import TallyFailedRequestHandler


@pytest.mark.parametrize('authentication_type', ['api_key', 'managed_identity', 'azureml_workspace_connection'])
@pytest.mark.skip('Need to mock calls for API key')
def test_create_success(authentication_type):
    """Test create success."""
    # Arrange
    configuration = ConfigurationParser().parse_configuration([
        '--authentication_type', authentication_type,
        '--scoring_url', 'hello.openai.azure.com'])
    tally_handler = TallyFailedRequestHandler(enabled=False)
    # Act
    scoring_client = AoaiScoringClient.create(configuration, tally_handler)

    # Assert
    assert isinstance(scoring_client, AoaiScoringClient)
