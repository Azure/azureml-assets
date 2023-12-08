# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scoring attempt."""

from .scoring_utils import RetriableType


class ScoringAttempt:
    """Scoring attempt."""

    def __init__(
        self,
        endpoint_base_url: str,
        response_status: int,
        model_response_code: str,
        retriable_type: RetriableType
    ):
        """Init function."""
        self.endpoint_base_url = endpoint_base_url
        self.response_status = response_status
        self.model_response_code = model_response_code
        self.retriable_type = retriable_type
