from enum import Enum

from ...common.telemetry import logging_utils as lu


class RoutingResponseType(Enum):
    RETRY = 1,
    USE_EXISTING = 2,
    FAILURE = 3,
    SUCCESS = 4


def classify_response(response_status: int) -> RoutingResponseType:
    if abs(response_status) == 200:
        classification = RoutingResponseType.SUCCESS
    elif abs(response_status) == 408 or response_status == -1:
        classification = RoutingResponseType.RETRY
    elif abs(response_status) >= 500:
        classification = RoutingResponseType.USE_EXISTING
    else:
        classification = RoutingResponseType.FAILURE

    lu.get_logger().debug(f"Response status of {response_status} was classified to {classification}.")
    return classification
