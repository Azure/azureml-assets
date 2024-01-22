# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for AOAI response handler."""

import asyncio

import aiohttp

from ...common.scoring.generic_scoring_client import (
    HttpResponseHandler,
    HttpScoringResponse,
)
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import (
    RetriableException,
    ScoringResult,
    ScoringResultStatus,
)
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry.events import event_utils
from ...common.telemetry.events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent


class AoaiHttpResponseHandler(HttpResponseHandler):
    """Defines the AOAI HTTP response handler."""

    RETRIABLE_STATUS_CODES = [408, 429, 500, 502, 503, 504]

    def __init__(self, tally_handler: TallyFailedRequestHandler):
        self.__tally_handler = tally_handler


    def handle_response(
        self,
        http_response: HttpScoringResponse,
        scoring_request: ScoringRequest,
        x_ms_client_request_id: str,
        start: float,
        end: float,
        worker_id: str
    ) -> ScoringResult:
        """ Handles the response from the model for the provided scoring request."""

        # Emit request completed event
        self._emit_request_completed_event(
            http_response=http_response,
            scoring_request=scoring_request,
            x_ms_client_request_id=x_ms_client_request_id,
            start=start,
            end=end,
            worker_id=worker_id
        )

        if http_response.exception:
            return self._handle_exception(
                http_response=http_response,
                scoring_request=scoring_request,
                start=start,
                end=end,
            )
        
        response_status = http_response.status
        if response_status == 200:
            return self._create_scoring_result(
                status=ScoringResultStatus.SUCCESS,
                scoring_request=scoring_request,
                start=start,
                end=end,
                http_post_response=http_response,
            )
        
        if self.is_retriable(response_status):
            raise RetriableException(
                status_code=http_response.status,
                response_payload=http_response.payload)

        result = self._create_scoring_result(
            status=ScoringResultStatus.FAILURE,
            scoring_request=scoring_request,
            start=start,
            end=end,
            http_post_response=http_response,
        )

        model_response_code = http_response.get_model_response_code()
        if (
            result.status == ScoringResultStatus.FAILURE
            and self.__tally_handler.should_tally(
                response_status=response_status,
                model_response_status=model_response_code)
        ):
            result.omit = True

        return result
    
    def is_retriable(self, http_status: int) -> bool:
        return http_status in self.RETRIABLE_STATUS_CODES

    def _handle_exception(
        self,
        http_response: HttpScoringResponse,
        scoring_request: ScoringRequest,
        start: float,
        end: float,
    ) -> ScoringResult:
        """ Handles the exception by raising retriable exception or creating scoring result for non-retriable exception."""
        try:
            raise http_response.exception
        except (
            aiohttp.ClientConnectorError,
            aiohttp.ServerConnectionError,
            asyncio.TimeoutError,
        ):
            raise RetriableException(
                status_code=http_response.status,
                response_payload=http_response.payload,
            )
        except Exception:
            return self._create_scoring_result(
                status=ScoringResultStatus.FAILURE,
                scoring_request=scoring_request,
                start=start,
                end=end,
                http_post_response=http_response,
            )
    
    def _emit_request_completed_event(
        self,
        http_response: HttpScoringResponse,
        scoring_request: ScoringRequest,
        x_ms_client_request_id: str,
        start: float,
        end: float,
        worker_id: str
    ) -> None:
        def get_prompt_tokens(response_body: any):
            if response_body and not isinstance(response_body, list):
                usage: dict[str, int] = response_body["usage"]
                return usage.get("prompt_tokens", None)
        
        def get_completion_tokens(response_body: any):
            if response_body and not isinstance(response_body, list):
                usage: dict[str, int] = response_body["usage"]
                return usage.get("completion_tokens", None)
            
        request_completed_event = BatchScoreRequestCompletedEvent(
            minibatch_id = scoring_request.mini_batch_context.mini_batch_id if scoring_request.mini_batch_context is not None else None,
            input_row_id = scoring_request.internal_id,
            x_ms_client_request_id = x_ms_client_request_id,
            worker_id = worker_id,
            scoring_url = scoring_request.scoring_url,
            is_successful = http_response.status == 200,
            is_retriable = self.is_retriable(http_response.status),
            response_code = http_response.status,
            model_response_code = http_response.get_model_response_code(),
            prompt_tokens = get_prompt_tokens(http_response.payload),
            completion_tokens = get_completion_tokens(http_response.payload),
            duration_ms = (end - start) * 1000,
            segmented_request_id = scoring_request.segment_id
        )
        event_utils.emit_event(batch_score_event = request_completed_event)