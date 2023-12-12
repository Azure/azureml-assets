# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quota estimator."""

from abc import ABC, abstractmethod


class QuotaEstimator(ABC):
    """Quota estimator."""

    @abstractmethod
    def estimate_request_cost(self, request_obj: any) -> int:
        """Estimate request cost."""
        pass

    @abstractmethod
    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        """Estimate response cost."""
        pass
