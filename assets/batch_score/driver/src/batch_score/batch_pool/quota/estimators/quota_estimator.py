from abc import ABC, abstractmethod


class QuotaEstimator(ABC):
    @abstractmethod
    def estimate_request_cost(self, request_obj: any) -> int:
        pass

    @abstractmethod
    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        pass
