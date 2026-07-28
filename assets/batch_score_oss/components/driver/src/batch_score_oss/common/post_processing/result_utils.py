# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Result utilities."""

from ..request_modification.input_transformer import InputTransformer
from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from .mini_batch_context import MiniBatchContext


def apply_input_transformer(
        input_to_output_transformer: InputTransformer,
        scoring_results: "list[ScoringResult]",
        mini_batch_context: MiniBatchContext = None):
    """Apply input to output transformation to the scoring results."""
    if input_to_output_transformer:
        lu.get_logger().debug("Start applying input to output transformer.")
        for scoring_result in scoring_results:
            # None request_obj values may be present, if a RequestModificationException
            # was thrown during ScoringRequest creation
            if not scoring_result.omit and scoring_result.request_obj:
                scoring_result.request_obj = input_to_output_transformer.apply_modifications(
                    request_obj=scoring_result.request_obj)

        lu.get_logger().debug("Completed input to output transform.")
    return scoring_results


def get_return_value(ret: 'list[str]', output_behavior: str):
    """Get return value according to the output behavior."""
    if (output_behavior == "summary_only"):
        lu.get_logger().info("Returning results in summary_only mode.")
        # PRS confirmed there is no way to allow users to toggle the output_action behavior in the v2 component.
        # A workaround is to return an empty array, but that can run afoul of the item-based error_threshold logic.
        # Instead, return an array of the same length as the results array, but with dummy values.
        # This is what core-search did in their fork of the component.
        return ["True"] * len(ret)

    lu.get_logger().info("Returning results in append_row mode.")
    return ret
