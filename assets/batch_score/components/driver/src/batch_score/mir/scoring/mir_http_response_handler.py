# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for MIR HTTP response handler."""

import asyncio

import aiohttp

from ...common.scoring import scoring_utils
from ...common.scoring.generic_scoring_client import (
    HttpResponseHandler,
    HttpScoringResponse
)
from ...common.scoring.scoring_attempt import ScoringAttempt
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import (
    RetriableException,
    ScoringResult,
    ScoringResultStatus
)
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.logging_utils import get_events_client
from ...utils.common import get_base_url


class MirHttpResponseHandler(HttpResponseHandler):
    """Defines the MIR HTTP response handler."""

    def __init__(self, tally_handler: TallyFailedRequestHandler):
        self.__tally_handler = tally_handler

    def handle_response(
            self,
            http_response: HttpScoringResponse,
            scoring_request: ScoringRequest,
            start: float,
            end: float,
            scoring_url: str) -> ScoringResult:
        """Handles the response from the model for the provided scoring request."""

        result: ScoringResult = None
        updated_http_response = self._update_http_response_for_exception(http_response)

        # Get values from http response
        response_status = updated_http_response.status
        response_payload = updated_http_response.payload
        model_response_code = updated_http_response.get_model_response_code()
        model_response_reason = updated_http_response.get_model_response_reason()

        retriable_type = scoring_utils.get_retriable_type(
            response_status=response_status,
            response_payload=response_payload,
            model_response_code=model_response_code,
            model_response_reason=model_response_reason
        )

        is_retriable = scoring_utils.is_retriable(retriable_type)

        endpoint_base_url = get_base_url(scoring_url)

        scoring_request.request_history.append(ScoringAttempt(
            endpoint_base_url=endpoint_base_url,
            response_status=response_status,
            model_response_code=model_response_code,
            retriable_type=retriable_type,
        ))

        get_events_client().emit_request_completed(
            latency=(end-start) * 1000,
            request_internal_id=scoring_request.internal_id,
            client_request_id=updated_http_response.headers["x-ms-client-request-id"],
            endpoint_uri=scoring_url,
            status_code="0" if response_status is None else str(response_status),
            model_response_code="" if model_response_code is None else model_response_code,
            client_exception=updated_http_response.exception_type,
            is_retriable=is_retriable
        )

        if response_status == 200:
            result = self._create_scoring_result(
                status=ScoringResultStatus.SUCCESS,
                scoring_request=scoring_request,
                start=start,
                end=end,
                http_post_response=updated_http_response,
                token_counts=scoring_request.estimated_token_counts,
                mini_batch_context=scoring_request.mini_batch_context
            )
            get_events_client().emit_tokens_generated(result.completion_tokens, result.prompt_tokens, scoring_url)
        elif is_retriable:
            raise RetriableException(
                status_code=response_status,
                response_payload=response_payload,
                model_response_code=model_response_code,
                model_response_reason=model_response_reason)
        else:  # Score failed
            result = self._create_scoring_result(
                status=ScoringResultStatus.FAILURE,
                scoring_request=scoring_request,
                start=start,
                end=end,
                http_post_response=updated_http_response,
                mini_batch_context=scoring_request.mini_batch_context
            )

            if self.__tally_handler.should_tally(
                response_status=response_status,
                model_response_status=model_response_code
            ):
                result.omit = True

        return result

    def _update_http_response_for_exception(
            self,
            http_response: HttpScoringResponse):
        if http_response.exception is None:
            return http_response  # No-op

        try:
            raise http_response.exception
        except (
            asyncio.TimeoutError,
            aiohttp.ServerConnectionError,
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError
        ):
            http_response.status = -408  # Manually attribute -408 as a tell to retry on this exception
        except Exception:
            http_response.status = -500  # Manually attribute -500 as a tell to not retry unhandled exceptions
        return http_response
