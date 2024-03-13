# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for abstract scoring client."""

from abc import abstractmethod

import aiohttp

from .scoring_request import ScoringRequest
from .scoring_result import ScoringResult


class ScoringClient:
    """Defines the scoring client."""

    @abstractmethod
    async def score(
        self,
        session: aiohttp.ClientSession,
        scoring_request: ScoringRequest,
        timeout: aiohttp.ClientTimeout = None,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score the request."""
        pass
