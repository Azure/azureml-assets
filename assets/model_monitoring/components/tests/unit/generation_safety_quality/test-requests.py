# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the request logic."""

from unittest.mock import MagicMock, patch, Mock
import pytest
import requests
from generation_safety_quality.annotation_compute_histogram.run import (
    _request_api)


@pytest.mark.unit
class TestRequests:
    """Test request logic."""

    @patch("requests.Session.post")
    def test_request_api_successful(
            self,
            mock_post):
        """Test request success."""
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "test response content"
                        },
                        "finish_reason": "test finish reason"
                    }
                ]
            }
        )

        token_manager = MagicMock()
        token_manager.get_token.return_value = "test_token"
        token_manager.auth_header = "Bearer"
        request_params = {
            "some_param": "some_value"
        }
        parsed_response, time_taken = _request_api(
            "azure_openai_api",
            requests.Session(),
            "https://www.example.org/test_endpoint",
            token_manager,
            **request_params,
        )

        assert time_taken >= 0
        assert parsed_response == {
            "samples": [
                "test response content"
            ],
            "finish_reason": [
                "test finish reason"
            ],
        }

    @patch("requests.Session.post")
    def test_request_api_fail(
            self,
            mock_post):
        """Test request failure with response code 500."""
        mock_post.return_value = Mock(
            status_code=500,
            text="test error message"
        )

        token_manager = MagicMock()
        token_manager.get_token.return_value = "test_token"
        token_manager.auth_header = "Bearer"
        request_params = {
            "some_param": "some_value"
        }
        with pytest.raises(Exception) as e:
            _request_api(
                "azure_openai_api",
                requests.Session(),
                "https://www.example.org/test_endpoint",
                token_manager,
                **request_params,
            )
        assert str(e.value) == \
            "Received unexpected HTTP status: 500 test error message"
