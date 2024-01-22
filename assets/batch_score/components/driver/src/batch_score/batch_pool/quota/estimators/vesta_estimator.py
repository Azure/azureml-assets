from .quota_estimator import QuotaEstimator


class VestaEstimator(QuotaEstimator):
    def estimate_request_cost(self, request_obj: any) -> int:
        return 1

    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        return 1