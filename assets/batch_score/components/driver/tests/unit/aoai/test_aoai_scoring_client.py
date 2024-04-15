# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for AOAI scoring client."""

from unittest.mock import MagicMock

from src.batch_score.aoai.scoring.aoai_response_handler import AoaiHttpResponseHandler
from src.batch_score.aoai.scoring.aoai_scoring_client import AoaiScoringClient


def test_init():
    """Test init."""
    # Arrange and Act
    scoring_client = AoaiScoringClient(
        header_provider=MagicMock(),
        scoring_url="https://scoring_url",
        tally_handler=MagicMock())

    # Assert
    assert scoring_client is not None
    assert type(scoring_client._http_response_handler) is AoaiHttpResponseHandler
