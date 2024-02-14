# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Scoring utilities."""

from enum import Enum


# If a traffic group is configured to exist on the endpoint,
# but there are zero deployments currently assigned to it,
# MIR frontdoor pretends the traffic group doesn't exist and throws this error
ZERO_TRAFFIC_GROUP_ERROR = "Specified traffic group could not be found"
DEFAULT_MAX_RETRIES = 3


class RetriableType(Enum):
    """Retriable type."""

    NOT_RETRIABLE = 1
    RETRY_ON_SAME_ENDPOINT = 2
    RETRY_ON_DIFFERENT_ENDPOINT = 3
    RETRY_UNTIL_MAX_RETRIES = 4


def is_retriable(retriable_type: RetriableType, retry_count: int, max_retries: int = DEFAULT_MAX_RETRIES):
    """Check whether the type is retriable."""
    return retriable_type != RetriableType.NOT_RETRIABLE \
        and not (retriable_type == RetriableType.RETRY_UNTIL_MAX_RETRIES and retry_count >= max_retries)


def is_zero_traffic_group_error(response_status: int, response_payload: any = None):
    """Check whether the response indicates a zero traffic group error."""
    return response_status == 404 and ZERO_TRAFFIC_GROUP_ERROR in response_payload


# 404 case is to exclude an endpoint that encountered ZeroTrafficGroupError
# 429, 503 case is to exclude an endpoint
# 424 case is to exclude an endpoint that encountered ModelNotReadyError
def get_retriable_type(
        response_status: int,
        response_payload: any = None,
        model_response_code: str = None,
        model_response_reason: str = None):
    """Get retriable type from response and model response."""
    if response_status in [408, -408]:
        return RetriableType.RETRY_ON_SAME_ENDPOINT

    if response_status in [429, 503]:
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if is_zero_traffic_group_error(response_status, response_payload):
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if response_status == 403:
        # TODO: Remove 403 from retriable statuses.
        # A bug in MIR returns 403 instead of 404 when allow-listed object ids exist
        # on traffic groups with zero capacity.
        return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if response_status == 424:
        if model_response_code in ["408", "504", "500"]:
            return RetriableType.RETRY_ON_SAME_ENDPOINT

        if model_response_code == "429":
            return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

        if model_response_code == "" and model_response_reason in ["model_not_ready", "too_few_model_instance"]:
            return RetriableType.RETRY_ON_DIFFERENT_ENDPOINT

    if response_status >= 500:
        return RetriableType.RETRY_UNTIL_MAX_RETRIES

    return RetriableType.NOT_RETRIABLE
