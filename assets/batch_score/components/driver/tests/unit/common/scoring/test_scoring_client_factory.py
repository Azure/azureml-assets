# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for scoring client factory."""

import pytest

from src.batch_score.aoai.scoring.aoai_scoring_client import AoaiScoringClient
from src.batch_score.batch_pool.scoring.scoring_client import ScoringClient
from src.batch_score.common.configuration.configuration_parser import ConfigurationParser
from src.batch_score.common.scoring.scoring_client_factory import ScoringClientFactory
from src.batch_score.common.scoring.tally_failed_request_handler import TallyFailedRequestHandler


@pytest.mark.parametrize('scoring_url, expected_scoring_client_type', [
    ('hello.openai.azure.com', AoaiScoringClient),
    ('servelessendpoint.inference.ai.azure.com', AoaiScoringClient),
    ('hello.centralus.inference.ml.azure.com', ScoringClient)
])
def test_create_success(mocker, make_metadata, scoring_url, expected_scoring_client_type):
    # Arrange
    configuration = ConfigurationParser().parse_configuration([
        '--scoring_url', scoring_url
    ])

    tally_handler = TallyFailedRequestHandler(enabled=False)

    # Act
    scoring_client = ScoringClientFactory().setup_scoring_client(configuration,
                                                                 make_metadata,
                                                                 tally_handler,
                                                                 routing_client=None)

    # Assert
    assert isinstance(scoring_client, expected_scoring_client_type)
