# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for MIR scoring client."""

import pytest
from unittest.mock import MagicMock

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.mir.scoring.mir_http_response_handler import MirHttpResponseHandler
from src.batch_score.mir.scoring.mir_scoring_client import MirScoringClient


url_1 = "https://scoring_url_1"
url_2 = "https://scoring_url_2"


@pytest.mark.parametrize(
    "scoring_url, configuration, expected_scoring_url",
    [
        # If scoring url is not provided, the configuration is used.
        (None, Configuration(scoring_url=url_2), url_2),

        # If scoring_url is provided, it overrides the configuration.
        (url_1, Configuration(scoring_url=url_2), url_1),
    ],
)
def test_init(scoring_url, configuration, expected_scoring_url):
    """Test init."""
    # Arrange and Act
    scoring_client = MirScoringClient(
        header_provider=MagicMock(),
        configuration=configuration,
        tally_handler=MagicMock(),
        scoring_url=scoring_url)

    # Assert
    assert scoring_client is not None
    assert type(scoring_client._http_response_handler) is MirHttpResponseHandler
    assert scoring_client._scoring_url == expected_scoring_url
