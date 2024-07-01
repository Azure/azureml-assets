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
from ...common.scoring.scoring_utils import get_prompt_tokens, get_completion_tokens
from ...common.telemetry.events import event_utils
from ...common.telemetry.events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent
from ...common.telemetry.logging_utils import get_events_client
from ...utils.common import get_base_url, get_mini_batch_id


class MirHttpResponseHandler(HttpResponseHandler):
    """Defines the MIR HTTP response handler."""

    def __init__(self, tally_handler: TallyFailedRequestHandler):
        """Initialize MirHttpResponseHandler."""
        self.__tally_handler = tally_handler

    def handle_response(
            self,
            scoring_request: ScoringRequest,
            http_response: HttpScoringResponse,
            x_ms_client_request_id: str,
            start: float,
            end: float,
            worker_id: str) -> ScoringResult:
        """Handle the response from the model for the provided scoring request."""
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

        if retriable_type == scoring_utils.RetriableType.RETRY_UNTIL_MAX_RETRIES:
            scoring_request.retry_count_for_limited_retries += 1

        is_retriable = scoring_utils.is_retriable(retriable_type, scoring_request.retry_count_for_limited_retries)

        endpoint_base_url = get_base_url(scoring_request.scoring_url)

        scoring_request.request_history.append(ScoringAttempt(
            endpoint_base_url=endpoint_base_url,
            response_status=response_status,
            model_response_code=model_response_code,
            retriable_type=retriable_type,
        ))

        get_events_client().emit_request_completed(
            latency=(end-start) * 1000,
            request_internal_id=scoring_request.internal_id,
            client_request_id=updated_http_response.headers.get("x-ms-client-request-id"),
            endpoint_uri=scoring_request.scoring_url,
            status_code="0" if response_status is None else str(response_status),
            model_response_code="" if model_response_code is None else model_response_code,
            client_exception=updated_http_response.exception_type,
            is_retriable=is_retriable
        )

        request_completed_event = BatchScoreRequestCompletedEvent(
            minibatch_id=get_mini_batch_id(scoring_request.mini_batch_context),
            input_row_id=scoring_request.internal_id,
            x_ms_client_request_id=x_ms_client_request_id,
            worker_id=worker_id,
            scoring_url=scoring_request.scoring_url,
            is_successful=response_status == 200,
            is_retriable=is_retriable,
            response_code=response_status,
            model_response_code=model_response_code,
            prompt_tokens=get_prompt_tokens(response_payload),
            completion_tokens=get_completion_tokens(response_payload),
            duration_ms=(end - start) * 1000,
            segmented_request_id=scoring_request.segment_id
        )
        event_utils.emit_event(batch_score_event=request_completed_event)

        if response_status == 200:
            result = self._create_scoring_result(
                status=ScoringResultStatus.SUCCESS,
<<<<<<< HEAD
=======
                model_response_code=response_status,
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
                scoring_request=scoring_request,
                start=start,
                end=end,
                http_post_response=updated_http_response,
                token_counts=scoring_request.estimated_token_counts,
                mini_batch_context=scoring_request.mini_batch_context
            )
            get_events_client().emit_tokens_generated(
                result.completion_tokens,
                result.prompt_tokens,
                scoring_request.scoring_url)
        elif is_retriable:
            raise RetriableException(
                status_code=response_status,
                response_payload=response_payload,
                model_response_code=model_response_code,
                model_response_reason=model_response_reason)
        else:  # Score failed
            result = self._create_scoring_result(
                status=ScoringResultStatus.FAILURE,
<<<<<<< HEAD
=======
                model_response_code=response_status,
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
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
