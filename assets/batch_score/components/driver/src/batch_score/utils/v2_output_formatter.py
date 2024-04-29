# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains the definition for the new (V2) output formatter.

V2 Output format:
{
    "custom_id": <custom_id>,
    "request_id": "", // MIR endpoint request id?
    "status": <HTTP response status code>,
    // If response is successful, "response" should have the response body and "error" should be null,
    // and vice versa for a failed response.
    "response": { <response_body> | null },
    "error": { null | <response_body> }
}
"""

from copy import deepcopy

from .output_formatter import OutputFormatter
from ..common.scoring.aoai_error import AoaiScoringError
from ..common.scoring.aoai_response import AoaiScoringResponse
from ..common.scoring.scoring_result import ScoringResult


class V2OutputFormatter(OutputFormatter):
    """Defines a class to format output in V2 format."""

    def __init__(self):
        """Initialize V2OutputFormatter."""
        self.estimator = None

    def format_output(self, results: "list[ScoringResult]", batch_size_per_request: int) -> "list[str]":
        """Format output in the V2 format."""
        output_list: list[dict[str, str]] = []
        for scoringResult in results:
            output: dict[str, str] = {}

            keys = scoringResult.request_obj.keys()
            if "custom_id" in keys:
                output["custom_id"] = scoringResult.request_obj["custom_id"]
            else:
                raise Exception("V2OutputFormatter called and custom_id not found in request object (original payload)")

            request_id = self.__get_request_id(scoringResult)
            status_code = scoringResult.model_response_code

            if scoringResult.status.name == "SUCCESS":
                response = AoaiScoringResponse(request_id=request_id,
                                               status_code=status_code,
                                               body=deepcopy(scoringResult.response_body))
                output["response"] = vars(response)
                output["error"] = None
            else:
                error = AoaiScoringError(message=deepcopy(scoringResult.response_body))
                output["response"] = {
                    "request_id": request_id,
                    "status_code": status_code,
                }
                output["error"] = vars(error)

            if batch_size_per_request > 1:
                # _convert_to_list_of_output_items() expects output["request"] to be set.
                output["request"] = scoringResult.request_obj
                batch_output_list = self._convert_to_list_of_output_items(
                    output,
                    scoringResult.estimated_token_counts)
                output_list.extend(batch_output_list)
            else:
                output_list.append(output)

        return list(map(self._stringify_output, output_list))

    def __get_request_id(self, scoring_request: ScoringResult):
        return scoring_request.response_headers.get("x-request-id", "")

    def _get_response_obj(self, result: dict):
        if result.get("response") is None:
            return result.get("error")
        else:
            return result.get("response").get("body")

    def _get_custom_id(self, result: dict) -> str:
        return result.get("custom_id", "")

    def _get_single_output(self, custom_id, result):
        single_output = {
            "id": "",  # TODO: populate this ID
            "custom_id": deepcopy(custom_id),
            "response": deepcopy(result["response"]),
            "error": deepcopy(result["error"])
        }
        return single_output
