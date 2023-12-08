# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vesta estimator."""

from .quota_estimator import QuotaEstimator


class VestaEstimator(QuotaEstimator):
    """Vesta estimator."""

    def estimate_request_cost(self, request_obj: any) -> int:
        """Estimate request cost."""
        return 1

    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        """Estimate response cost."""
        return 1
