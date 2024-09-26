# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for null endpoint scoring client."""

import time

import aiohttp

from .http_response_handler import HttpResponseHandler
from .http_scoring_response import HttpScoringResponse
from .scoring_request import ScoringRequest
from .scoring_result import ScoringResult

from ..telemetry.scoring_logging import (
    ScoreStartLog,
    ScoreSucceedLog,
)


class NullEndpointScoringClient:
    """Defines the null endpoint scoring client."""

    def __init__(
        self,
        http_response_handler: HttpResponseHandler,
        scoring_url: str,
    ) -> None:
        """Initialize NullEndpointScoringClient."""
        self._http_response_handler = http_response_handler
        self._scoring_url = scoring_url

    async def score(
        self,
        session: aiohttp.ClientSession,
        scoring_request: ScoringRequest,
        timeout: aiohttp.ClientTimeout = None,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score the request against the scoring url."""
        scoring_request.scoring_url = self._scoring_url

        start = time.time()

        ScoreStartLog(
            worker_id=worker_id,
            internal_id=scoring_request.internal_id,
            x_ms_client_request_id='',
            scoring_url=self._scoring_url,
            timeout=timeout
        ).log()

        http_response: HttpScoringResponse = self._get_mock_response()

        end = time.time()
        if http_response.status == 200:
            # Clean up this log event
            ScoreSucceedLog(
                worker_id=worker_id,
                internal_id=scoring_request.internal_id,
                x_ms_client_request_id='',
                scoring_url=self._scoring_url,
                duration=end-start
            ).log()

        scoring_result: ScoringResult = self._http_response_handler.handle_response(
            scoring_request=scoring_request,
            http_response=http_response,
            x_ms_client_request_id='',
            start=start,
            end=end,
            worker_id=worker_id
        )

        return scoring_result

    def _get_mock_response(self):
        return HttpScoringResponse(
            status=200,
            headers={},
            payload={
                "id": "chatcmpl-mock-request-id",
                "object": "chat.completion",
                "created": 1714932323,
                "model": "gpt-4",
                "system_fingerprint": "fp_mock",
                "usage": {
                        "completion_tokens": 35,
                        "prompt_tokens": 223,
                        "total_tokens": 258
                    },
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {
                            "content": "a brown fox jumps over the lazy dog",
                            "role": "assistant"
                        }
                    }
                ]
            }
        )
