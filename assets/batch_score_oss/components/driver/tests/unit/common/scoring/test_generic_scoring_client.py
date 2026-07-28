# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for generic scoring client."""

import aiohttp
import pytest
from unittest.mock import MagicMock, patch

from src.batch_score_oss.common.scoring.generic_scoring_client import GenericScoringClient
from src.batch_score_oss.common.scoring.scoring_request import ScoringRequest
from src.batch_score_oss.common.request_modification.modifiers.input_type_modifier import InputTypeModifier
from src.batch_score_oss.common.request_modification.input_transformer import InputTransformer
from src.batch_score_oss.common.common_enums import InputType

from tests.fixtures.client_response import FakeResponse


class NullHeaderProvider:
    """Null header provider."""

    def get_headers(self):
        """Get headers."""
        return {"static_header": "static_value"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_status, response_body, exception_to_raise",
    [
        (200, {"response": "Test response"}, None),
        (400, None, None),
        (None, None, Exception),
    ],
)
async def test_score(response_status, response_body, exception_to_raise):
    """Test score once."""
    # Arrange
    http_response_handler = MagicMock()
    http_response_handler.handle_response.return_value = MagicMock()
    scoring_client = GenericScoringClient(
        header_provider=NullHeaderProvider(),
        http_response_handler=http_response_handler,
        scoring_url=None)

    input_type_modifier = InputTypeModifier()
    input_transfomer = InputTransformer([input_type_modifier])
    scoring_request = ScoringRequest(
        original_payload='{"custom_id": "task_123", "messages": [{"content": "just text"}]}',
        input_to_request_transformer=input_transfomer,
    )

    async with aiohttp.ClientSession() as session:
        with patch.object(session, "post") as mock_post:
            if exception_to_raise:
                mock_post.return_value.__aenter__.side_effect = exception_to_raise
            else:
                mock_post.return_value.__aenter__.return_value = FakeResponse(
                    status=response_status,
                    json=response_body)

            # Act
            result = await scoring_client.score(
                session=session,
                scoring_request=scoring_request,
                timeout=None,
                worker_id="1")

    # Assert
    assert scoring_request.input_type == InputType.TextOnly
    assert http_response_handler.handle_response.assert_called_once
    response_sent_to_handler = http_response_handler.handle_response.call_args.kwargs['http_response']

    generated_request = scoring_client._create_http_request(scoring_request)
    assert "custom_id" not in generated_request.payload
    assert "input_type" not in generated_request.payload
    assert generated_request.headers["static_header"] == "static_value"

    if exception_to_raise:
        assert type(response_sent_to_handler.exception) is exception_to_raise
    elif response_status == 200:
        assert response_sent_to_handler.status == response_status
        assert response_sent_to_handler.payload == response_body
    else:
        assert response_sent_to_handler.status == response_status
        assert response_sent_to_handler.payload == ''

    assert result == http_response_handler.handle_response.return_value


@pytest.mark.parametrize(
    "response_status, exeception_raised_while_scoring, exception_raised_by_validate_auth",
    [
        # Raise exception for 401 and 403 response.
        (401, None, Exception),
        (403, None, Exception),
        # Do not raise an exception otherwise.
        (200, None, None),
        (400, None, None),
        (None, Exception, None),
    ],
)
def test_validate_auth(
        response_status,
        exeception_raised_while_scoring,
        exception_raised_by_validate_auth):
    """Test validate auth."""
    # Arrange
    scoring_client = GenericScoringClient(
        header_provider=NullHeaderProvider(),
        http_response_handler=MagicMock(),
        scoring_url=None)

    async def mock_invalid_pool_routes(*args, **kwargs):
        if exeception_raised_while_scoring:
            raise exeception_raised_while_scoring

        return FakeResponse(
            status=response_status,
            json=None)

    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_post.return_value.__aenter__.side_effect = mock_invalid_pool_routes

        # Act and Assert
        if not exception_raised_by_validate_auth:
            scoring_client.validate_auth()
        else:
            with pytest.raises(exception_raised_by_validate_auth) as exc_info:
                scoring_client.validate_auth()

            assert "Scoring Client auth check failed." in str(exc_info.value)
