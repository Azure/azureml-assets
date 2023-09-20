# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Method to check if the response is retriable."""

# If a traffic group is configured to exist on the endpoint,
# but there are zero deployments currently assigned to it,
# MIR frontdoor pretends the traffic group doesn't exist and throws this error
zero_traffic_group_error = "Specified traffic group could not be found"


def is_retriable(
        response_status: int, response_payload: any = None,
        model_response_code: str = None, model_response_reason: str = None
) -> bool:
    """Check if the response is retriable."""
    model_response_code = None if model_response_code == "" else model_response_code
    model_response_reason = None if model_response_reason == "" else model_response_reason

    if response_status in [429, 408, -408, 503]:
        return True
    elif response_status == 424:
        if model_response_code is None and (model_response_reason == "model_not_ready"
                                            and isinstance(response_payload, str) and
                                            response_payload == "no healthy upstream"):
            return True
        elif any(model_response_code == x for x in ["429", "408", "504", "500"]):
            return True
    elif response_status == 404 and zero_traffic_group_error in response_payload:
        return True
    else:
        return False
