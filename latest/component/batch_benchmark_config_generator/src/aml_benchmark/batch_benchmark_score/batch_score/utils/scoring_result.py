# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for scoring results."""

from multidict import CIMultiDictProxy
from enum import Enum
from . import logging_utils as lu


class RetriableException(Exception):
    """Retriable exception class."""

    def __init__(
            self, status_code: int, response_payload: any = None,
            model_response_code: str = None, model_response_reason: str = None,
            retry_after: float = None
    ):
        """Init method."""
        self.status_code = status_code
        self.response_payload = response_payload
        self.model_response_code = model_response_code
        self.model_response_reason = model_response_reason
        self.retry_after = retry_after


class ScoringResultStatus(Enum):
    """Enum for scroing results."""

    FAILURE = 1
    SUCCESS = 2


class ScoringResult:
    """Scoring result class."""

    def __init__(
            self, status: ScoringResultStatus, start: float, end: float, request_obj: any,
            request_metadata: any, response_body: any,
            response_headers: CIMultiDictProxy[str], num_retries: int, omit: bool = False
    ):
        """Init method."""
        self.status = status
        self.start = start
        self.end = end
        self.request_obj = request_obj  # Normalize to json
        self.request_metadata = request_metadata
        self.response_body = response_body
        self.response_headers = response_headers
        self.num_retries = num_retries
        self.omit = omit
        self.segmented_response_bodies: list[any] = None

        self.duration = end - start
        self.prompt_tokens = None
        self.completion_tokens = None
        self.total_tokens = None

        self.__analyze()

    def __analyze(self):
        try:
            if self.status == ScoringResultStatus.FAILURE:
                return

            # usage: dict[str, int] = self.response_body["usage"]
            if len(self.response_body) < 1:
                return
            usage: dict[str, int] = self.response_body
            if usage is None:
                return
            if isinstance(usage, list):
                usage = usage[0]
                if '0' in usage.keys():
                    usage = usage['0']

            # self.prompt_tokens = usage.get("prompt_tokens", None)
            # self.completion_tokens = usage.get("completion_tokens", None)
            # self.total_tokens = usage.get("total_tokens", None)
        except ValueError as e:
            print(e)
            lu.get_logger().error("response is not a json")
