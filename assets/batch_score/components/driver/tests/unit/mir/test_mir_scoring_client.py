# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for MIR scoring client."""

import pytest
from unittest.mock import MagicMock, patch

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
    scoring_client = _get_scoring_client(
        configuration=configuration,
        scoring_url=scoring_url)

    # Assert
    assert scoring_client is not None
    assert scoring_client._generic_scoring_client is not None
    assert type(scoring_client._generic_scoring_client._http_response_handler) is MirHttpResponseHandler
    assert scoring_client._generic_scoring_client._scoring_url == expected_scoring_url


@pytest.mark.asyncio
async def test_score_once():
    """Test score_once."""
    # Arrange
    scoring_client = _get_scoring_client()
    with patch.object(scoring_client._generic_scoring_client, "score") as mock_score_once:
        mock_score_once.return_value = MagicMock()

        # Act
        result = await scoring_client.score_once(
            session=MagicMock(),
            scoring_request=MagicMock(),
            timeout=MagicMock(),
            worker_id="1")

        # Assert
        assert mock_score_once.assert_called_once
        assert result == mock_score_once.return_value


def test_validate_auth():
    """Test validate_auth."""
    # Arrange
    scoring_client = _get_scoring_client()
    with patch.object(scoring_client._generic_scoring_client, "validate_auth") as mock_validate_auth:
        # Act
        scoring_client.validate_auth()

        # Assert
        assert mock_validate_auth.assert_called_once


def _get_scoring_client(configuration=None, scoring_url=None):
    return MirScoringClient(
        header_provider=MagicMock(),
        configuration=configuration or Configuration(),
        tally_handler=MagicMock(),
        scoring_url=scoring_url)
