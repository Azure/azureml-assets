# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for abstract HTTP response handler."""

from abc import abstractmethod

from ..post_processing.mini_batch_context import MiniBatchContext
from .http_scoring_response import HttpScoringResponse
from .scoring_request import ScoringRequest
from .scoring_result import (
    ScoringResult,
    ScoringResultStatus
)


class HttpResponseHandler:
    @abstractmethod
    def handle_response(
            self,
            scoring_request: ScoringRequest,
            http_response: HttpScoringResponse,
            x_ms_client_request_id: str,
            start: float,
            end: float,
            worker_id: str) -> ScoringResult:
        pass

    def _create_scoring_result(
            self,
            status: ScoringResultStatus,
            scoring_request: ScoringRequest,
            start: float,
            end: float,
            http_post_response: HttpScoringResponse,
            token_counts: "tuple[int]" = (),
            mini_batch_context: MiniBatchContext = None) -> ScoringResult:
        """ Creates an instance of scoring result."""
        return ScoringResult(
            status=status,
            start=start,
            end=end,
            request_obj=scoring_request.original_payload_obj,
            request_metadata=scoring_request.request_metadata,
            response_body=http_post_response.payload,
            response_headers=http_post_response.headers,
            num_retries=0,
            token_counts=token_counts,
            mini_batch_context=mini_batch_context
        )
