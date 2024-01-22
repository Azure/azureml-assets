# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for generic scoring client."""

import asyncio
import json
import time
import traceback
import uuid

import aiohttp

from datetime import timedelta
from .header_provider import HeaderProvider
from .http_response_handler import HttpResponseHandler
from .http_scoring_request import HttpScoringRequest
from .http_scoring_response import HttpScoringResponse
from .scoring_request import ScoringRequest
from .scoring_result import ScoringResult

from ..telemetry.scoring_logging import (
    ScoreFailedLog,
    ScoreFailedWithExceptionLog,
    ScoreStartLog,
    ScoreSucceedLog,
)


class GenericScoringClient:
    """Defines the generic scoring client."""

    def __init__(
        self,
        header_provider: HeaderProvider,
        http_response_handler: HttpResponseHandler,
        scoring_url: str,
    ) -> None:
        self._header_provider = header_provider
        self._http_response_handler = http_response_handler
        self._scoring_url = scoring_url

    async def score(
        self,
        session: aiohttp.ClientSession,
        scoring_request: ScoringRequest,
        timeout: aiohttp.ClientTimeout = None,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Scores the request against the scoring url."""
        scoring_request.scoring_url = self._scoring_url
        http_request: HttpScoringRequest = self._create_http_request(scoring_request)

        start = time.time()

        ScoreStartLog(
            internal_id=scoring_request.internal_id,
            x_ms_client_request_id=http_request.headers['x-ms-client-request-id'],
            scoring_url=self._scoring_url
        ).log()

        http_response: HttpScoringResponse = await self._send_http_request(
            session=session,
            http_scoring_request=http_request,
            internal_id=scoring_request.internal_id,
            timeout=timeout,
        )

        end = time.time()
        if http_response.status == 200:
            # Clean up this log event
            ScoreSucceedLog(
                internal_id=scoring_request.internal_id,
                x_ms_client_request_id=http_request.headers["x-ms-client-request-id"],
                scoring_url=self._scoring_url,
                duration=end-start
            ).log()

        scoring_result: ScoringResult = self._http_response_handler.handle_response(
            scoring_request=scoring_request,
            http_response=http_response,
            x_ms_client_request_id=http_request.headers['x-ms-client-request-id'],
            start=start,
            end=end,
            worker_id=worker_id
        )

        return scoring_result

    def validate_auth(self):
        """Validates the auth by sending dummy request to the scoring url."""
        asyncio.run(self._validate_auth())

    async def _validate_auth(self):
        http_request = HttpScoringRequest(
            headers=self._header_provider.get_headers(),
            payload='hello',
            url=self._scoring_url,
        )
        """Validates the auth by sending dummy request to the scoring url."""
        timeout_duration = timedelta(seconds=10).total_seconds()
        timeout = aiohttp.ClientTimeout(total=timeout_duration)

        async with aiohttp.ClientSession() as session:
            http_response: HttpScoringResponse = await self._send_http_request(
                session=session,
                http_scoring_request=http_request,
                internal_id=str(uuid.uuid4()),
                timeout=timeout,
            )

        if http_response.status in [401, 403]:
            raise Exception(f"Scoring Client auth check failed. Error: {json.dumps(http_response.payload)}")

    def _create_http_request(
        self,
        scoring_request: ScoringRequest,
    ) -> HttpScoringRequest:
        """Creates an instance of HTTP scoring request."""
        return HttpScoringRequest(
            headers=self._header_provider.get_headers(),
            payload=scoring_request.cleaned_payload,
            url=self._scoring_url,
        )

    async def _send_http_request(
        self,
        session: aiohttp.ClientSession,
        http_scoring_request: HttpScoringRequest,
        internal_id: str,
        timeout: aiohttp.ClientTimeout = None,
    ) -> HttpScoringResponse:
        """Sends HTTP request to scoring url."""
        try: 
            async with session.post(
                url=http_scoring_request.url,
                headers=http_scoring_request.headers,
                data=http_scoring_request.payload,
                timeout=timeout,
            ) as response:
                return await self._log_failure_and_create_http_scoring_response(
                    response,
                    internal_id,
                    http_scoring_request.headers['x-ms-client-request-id'])
        except Exception as ex:
            return self._log_and_create_http_scoring_response_from_exception(
                ex,
                internal_id,
                http_scoring_request.headers['x-ms-client-request-id'])

    async def _log_failure_and_create_http_scoring_response(
        self,
        response: aiohttp.ClientResponse,
        internal_id: str,
        x_ms_client_request_id: str
    ) -> HttpScoringResponse:
        """Creates an instance of HTTP scoring response."""
        http_response = HttpScoringResponse(
            status=response.status,
            headers=response.headers,
        )

        if http_response.status == 200:
            http_response.payload = await response.json()
            return http_response

        error_response = await response.text()
        ScoreFailedLog(
            internal_id=internal_id,
            x_ms_client_request_id=x_ms_client_request_id,
            scoring_url=self._scoring_url,
            status_code="NONE" if response is None else response.status,
            reason="NONE" if response is None else response.reason,
            response_headers="NONE" if response is None else response.headers,
            response_payload=error_response
        ).log()

        try:
            http_response.payload = json.loads(error_response)
        except Exception:
            http_response.payload = error_response

        return http_response

    def _log_and_create_http_scoring_response_from_exception(
        self,
        ex: Exception,
        internal_id: str,
        x_ms_client_request_id: str
    ) -> HttpScoringResponse:
        """Logs failure and creates an instance of HTTP scoring response."""
        error = traceback.format_exc()
        ScoreFailedWithExceptionLog(
            internal_id=internal_id,
            x_ms_client_request_id=x_ms_client_request_id,
            scoring_url=self._scoring_url,
            exception_type=type(ex).__name__,
            exception=error,
            unhandled_exc=True
        ).log()

        # Todo: Pass headers here - required by some events.
        return HttpScoringResponse(
            exception=ex,
            exception_traceback=error,
            exception_type=type(ex).__name__
        )
