import os

from ..request_modification.input_transformer import InputTransformer
from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from .mini_batch_context import MiniBatchContext


def apply_input_transformer(
        input_to_output_transformer: InputTransformer,
        scoring_results: "list[ScoringResult]",
        mini_batch_context: MiniBatchContext = None):
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
    if (output_behavior == "summary_only"):
        lu.get_logger().info("Returning results in summary_only mode.")
        # PRS confirmed there is no way to allow users to toggle the output_action behavior in the v2 component.
        # A workaround is to return an empty array, but that can run afoul of the item-based error_threshold logic.
        # Instead, return an array of the same length as the results array, but with dummy values.
        # This is what core-search did in their fork of the component.
        return ["True"] * len(ret)

    lu.get_logger().info("Returning results in append_row mode.")
    return ret


def save_mini_batch_results(mini_batch_results: list, mini_batch_results_out_directory: str, raw_mini_batch_context):
    lu.get_logger().debug("mini_batch_results_out_directory: {}".format(mini_batch_results_out_directory))

    filename = f"{raw_mini_batch_context.minibatch_index}.jsonl"
    file_path = os.path.join(mini_batch_results_out_directory, filename)

    lu.get_logger().debug(f"Start saving {len(mini_batch_results)} results to file {file_path}.")
    with open(file_path, "w", encoding="utf-8") as writer:
        for item in mini_batch_results:
            writer.write(item + "\n")

    lu.get_logger().info(f"Completed saving {len(mini_batch_results)} results to file {file_path}.")
