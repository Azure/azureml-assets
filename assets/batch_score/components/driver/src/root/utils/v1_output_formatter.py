# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains the definition for the original (V1) output formatter.

V1 Output format:
{
    "status": ["SUCCESS" | "FAILURE"],
    "start": 1709584163.2691997,
    "end": 1709584165.2570084,
    "request": { <request_body> },
    "response": { <response_body> }
}
"""

from copy import deepcopy

from .output_formatter import OutputFormatter
from ..common.scoring.scoring_result import ScoringResult


class V1OutputFormatter(OutputFormatter):
    """Defines a class to format output in V1 format."""

    def __init__(self):
        """Initialize V1OutputFormatter."""
        self.estimator = None

    def format_output(self, results: "list[ScoringResult]", batch_size_per_request: int) -> "list[str]":
        """Format output in the V1 format."""
        output_list: list[dict[str, str]] = []
        for scoringResult in results:
            output: dict[str, str] = {}
            output["status"] = scoringResult.status.name
            output["start"] = scoringResult.start
            output["end"] = scoringResult.end
            output["request"] = scoringResult.request_obj
            output["response"] = scoringResult.response_body

            if scoringResult.segmented_response_bodies is not None and \
               len(scoringResult.segmented_response_bodies) > 0:
                output["segmented_responses"] = scoringResult.segmented_response_bodies

            if scoringResult.request_metadata is not None:
                output["request_metadata"] = scoringResult.request_metadata

            if batch_size_per_request > 1:
                batch_output_list = self._convert_to_list_of_output_items(
                    output,
                    scoringResult.estimated_token_counts)
                output_list.extend(batch_output_list)
            else:
                output_list.append(output)

        return list(map(self._stringify_output, output_list))

    def _get_response_obj(self, result: dict):
        return result["response"]

    def _get_custom_id(self, result: dict):
        return None

    def _get_request_id(self, result: dict):
        return None

    def _get_status(self, result: dict):
        return result["status"]

    def _get_single_output(self, request, response_obj, input_batch):
        single_output = {"request": deepcopy(request), "response": deepcopy(response_obj)}
        single_output["request"]["input"] = input_batch

        return single_output
