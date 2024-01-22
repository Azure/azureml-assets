from enum import Enum


# If a traffic group is configured to exist on the endpoint,
# but there are zero deployments currently assigned to it,
# MIR frontdoor pretends the traffic group doesn't exist and throws this error
ZeroTrafficGroupError = "Specified traffic group could not be found"


class RetriableType(Enum):
    NOT_RETRIABLE = 1
    RETRY_ON_SAME_ENDPOINT = 2
    RETRY_ON_DIFFERENT_ENDPOINT = 3


def is_retriable(retriable_type: RetriableType):
    return retriable_type != RetriableType.NOT_RETRIABLE


def is_zero_traffic_group_error(response_status: int, response_payload: any = None):
    return response_status == 404 and ZeroTrafficGroupError in response_payload


# 404 case is to exclude an endpoint that encountered ZeroTrafficGroupError
# 429, 503 case is to exclude an endpoint
# 424 case is to exclude an endpoint that encountered ModelNotReadyError
def get_retriable_type(response_status: int, response_payload: any = None, model_response_code: str = None, model_response_reason: str = None):
    if response_status in [408, -408]:
        return RetriableType.RETRY_ON_SAME_ENDPOINT

    if response_status in [429, 503]:
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if is_zero_traffic_group_error(response_status, response_payload):
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if response_status == 403:
        # TODO: Remove 403 from retriable statuses.
        #  A bug in MIR returns 403 instead of 404 when allow-listed object ids exist on traffic groups with zero capacity.
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if response_status == 424:
        if model_response_code in ["408", "504", "500"]:
            return RetriableType.RETRY_ON_SAME_ENDPOINT

        if model_response_code == "429":
            return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

        if model_response_code == "" and model_response_reason in ["model_not_ready", "too_few_model_instance"]:
            return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    return RetriableType.NOT_RETRIABLE