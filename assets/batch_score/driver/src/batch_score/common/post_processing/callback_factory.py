# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import traceback

from ...utils.common import convert_result_list
from ..configuration.configuration import Configuration
from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from ..telemetry.logging_utils import set_mini_batch_id
from .mini_batch_context import MiniBatchContext
from .result_utils import (
    apply_input_transformer,
    get_return_value,
    save_mini_batch_results,
)


def add_callback(callback, cur):
    def wrapper(scoring_results: "list[ScoringResult]", mini_batch_context: MiniBatchContext):
        scoring_results = callback(scoring_results, mini_batch_context)
        return cur(scoring_results, mini_batch_context)
    return wrapper


class CallbackFactory:
    def __init__(self,
                 configuration: Configuration,
                 input_to_output_transformer):
        self._configuration = configuration
        self.__input_to_output_transformer = input_to_output_transformer

    def generate_callback(self):
        callback = self.save_mini_batch_result_and_emit
        callback = add_callback(self.convert_result_list, callback)
        callback = add_callback(self.apply_input_transformer, callback)
        return callback

    def convert_result_list(self, scoring_results: "list[ScoringResult]", mini_batch_context: MiniBatchContext):
        return convert_result_list(results=scoring_results,
                                   batch_size_per_request=self._configuration.batch_size_per_request)

    def apply_input_transformer(self, scoring_results: "list[ScoringResult]", mini_batch_context: MiniBatchContext):
        return apply_input_transformer(self.__input_to_output_transformer, scoring_results)

    def save_mini_batch_result_and_emit(self,
                                        scoring_results: "list[ScoringResult]",
                                        mini_batch_context: MiniBatchContext):
        mini_batch_id = mini_batch_context.mini_batch_id
        set_mini_batch_id(mini_batch_context.mini_batch_id)
        lu.get_logger().info("Start saving result of data subset {}.".format(mini_batch_id))
        try:
            if mini_batch_context.exception is None:
                if self._configuration.save_mini_batch_results == "enabled":
                    lu.get_logger().info("save_mini_batch_results is enabled")
                    save_mini_batch_results(
                        scoring_results,
                        self._configuration.mini_batch_results_out_directory,
                        mini_batch_context.raw_mini_batch_context)
                else:
                    lu.get_logger().info("save_mini_batch_results is disabled")

                lu.get_events_client().emit_mini_batch_completed(
                    input_row_count=mini_batch_context.target_result_len,
                    output_row_count=len(scoring_results))
            else:
                ex = mini_batch_context.exception
                lu.get_events_client().emit_mini_batch_completed(
                    input_row_count=mini_batch_context.target_result_len,
                    output_row_count=len(scoring_results),
                    exception=type(ex).__name__,
                    stacktrace=traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
        except Exception as e:
            lu.get_events_client().emit_mini_batch_completed(
                input_row_count=mini_batch_context.target_result_len,
                output_row_count=len(scoring_results),
                exception=type(e).__name__,
                stacktrace=traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))

        lu.get_logger().info("Completed data subset {}.".format(mini_batch_id))
        set_mini_batch_id(None)

        return get_return_value(scoring_results, self._configuration.output_behavior)
