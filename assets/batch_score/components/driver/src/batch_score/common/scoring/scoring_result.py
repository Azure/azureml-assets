# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scoring result."""

from copy import deepcopy
from enum import Enum

from multidict import CIMultiDictProxy

from ..post_processing.mini_batch_context import MiniBatchContext
from ..telemetry import logging_utils as lu
from .scoring_request import ScoringRequest


class PermanentException(Exception):
    """Permanent exception."""

    def __init__(self, message: str, status_code: int = None, response_payload: any = None):
        """Initialize PermanentException."""
        super().__init__(message)

        self.status_code = status_code
        self.response_payload = response_payload


class RetriableException(Exception):
    """Retriable exception."""

    def __init__(
            self,
            status_code: int,
            response_payload:
            any = None,
            model_response_code: str = None,
            model_response_reason: str = None,
            retry_after: float = None):
        """Initialize RetriableException."""
        self.status_code = status_code
        self.response_payload = response_payload
        self.model_response_code = model_response_code
        self.model_response_reason = model_response_reason
        self.retry_after = retry_after


class ScoringResultStatus(Enum):
    """Scoring result status."""

    FAILURE = 1
    SUCCESS = 2


class ScoringResult:
    """Scoring result."""

    def __init__(
            self,
            status: ScoringResultStatus,
            start: float,
            end: float,
            request_obj: any,
            request_metadata: any,
            response_body: any,
            response_headers: CIMultiDictProxy[str],
            num_retries: int,
            omit: bool = False,
            token_counts: "tuple[int]" = (),
            mini_batch_context: MiniBatchContext = None):
        """Initialize ScoringResult."""
        self.status = status
        self.start = start
        self.end = end
        self.request_obj = request_obj  # Normalize to json
        self.request_metadata = request_metadata
        self.response_body = response_body
        self.response_headers = response_headers
        self.num_retries = num_retries
        self.omit = omit
        self.mini_batch_context: MiniBatchContext = mini_batch_context
        self.segmented_response_bodies: list[any] = None

        self.duration = end - start
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None

        self.__token_counts = token_counts

        self.__analyze()

    # read-only
    @property
    def estimated_token_counts(self) -> "tuple[int]":
        """Get the estimated token count."""
        return self.__token_counts

    def __analyze(self):
        try:
            if self.status == ScoringResultStatus.FAILURE:
                return

            if not isinstance(self.response_body, list):
                usage: dict[str, int] = self.response_body["usage"]

                self.prompt_tokens = usage.get("prompt_tokens", None)
                self.completion_tokens = usage.get("completion_tokens", None)
                self.total_tokens = usage.get("total_tokens", None)
        except ValueError:
            lu.get_logger().error("response is not a json")

    def Failed(scoring_request: ScoringRequest = None) -> 'ScoringResult':
        """Get a scoring result of failed status."""
        return ScoringResult(
            status=ScoringResultStatus.FAILURE,
            start=0,
            end=0,
            request_obj=scoring_request.original_payload_obj if scoring_request else None,
            request_metadata=scoring_request.request_metadata if scoring_request else None,
            response_body=None,
            response_headers=None,
            num_retries=scoring_request.retry_count if scoring_request else 0,
            omit=False,
            mini_batch_context=scoring_request.mini_batch_context if scoring_request else None
        )

    def copy(self) -> 'ScoringResult':
        """Return a copy of this result with a deep copy of response body and unset token counts."""
        """
        Deepcopy the response_body dictionary, as this will be edited in the final result of segmented responses.
        Set the three token attributes to None so others are not confused by the previous values.
        """
        copied_self = ScoringResult(
            self.status,
            self.start,
            self.end,
            self.request_obj,
            self.request_metadata,
            deepcopy(self.response_body),
            self.response_headers,
            self.num_retries,
            self.omit,
            self.__token_counts,
            self.mini_batch_context
        )
        copied_self.prompt_tokens = None
        copied_self.completion_tokens = None
        copied_self.total_tokens = None
        copied_self.segmented_response_bodies = self.segmented_response_bodies
        return copied_self
