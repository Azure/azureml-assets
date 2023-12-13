# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parallel driver."""

import asyncio

from ...utils.common import convert_result_list
from ..configuration.configuration import Configuration
from ..post_processing.mini_batch_context import MiniBatchContext
from ..post_processing.result_utils import apply_input_transformer
from ..request_modification.input_transformer import InputTransformer
from ..request_modification.modifiers.request_modifier import (
    RequestModificationException,
)
from ..scoring.scoring_request import ScoringRequest
from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from .conductor import Conductor


class Parallel:
    """Parallel driver."""

    def __init__(
        self,
        configuration: Configuration,
        loop: asyncio.AbstractEventLoop,
        conductor: Conductor,
        input_to_request_transformer: InputTransformer = None,
        input_to_log_transformer: InputTransformer = None,
        input_to_output_transformer: InputTransformer = None,
    ):
        """Init function."""
        self._configuration = configuration
        self.__loop = loop
        self.__conductor: Conductor = conductor
        self.__input_to_request_transformer: InputTransformer = input_to_request_transformer
        self.__input_to_log_transformer: InputTransformer = input_to_log_transformer
        self.__input_to_output_transformer: InputTransformer = input_to_output_transformer

    def run(self, payloads: "list[str]", mini_batch_context: MiniBatchContext = None):
        """Run function. Used in sync mode only."""
        self._log_start(payloads)
        scoring_requests, scoring_results = self._generate_scoring_requests(payloads, mini_batch_context)

        scoring_results.extend(self.__loop.run_until_complete(self.__conductor.run(scoring_requests)))

        apply_input_transformer(self.__input_to_output_transformer, scoring_results)

        results = convert_result_list(
            results=scoring_results,
            batch_size_per_request=self._configuration.batch_size_per_request)

        return results

    def enqueue(self, payloads: "list[str]", mini_batch_context: MiniBatchContext = None):
        """Enqueue function. Used in async mode only."""
        self._log_start(payloads)
        scoring_requests, failed_scoring_results = self._generate_scoring_requests(payloads, mini_batch_context)

        self.__conductor.enqueue(scoring_requests, failed_scoring_results, mini_batch_context)

    def check_all_tasks_processed(self):
        """Check all tasks processed. Used in async mode only."""
        return self.__conductor.check_all_tasks_processed()

    def get_finished_batch_result(self):
        """Get finished batch result. Used in async mode only."""
        return self.__conductor.get_finished_batch_result()

    def get_processing_batch_number(self):
        """Get processing batch number. Used in async mode only."""
        return self.__conductor.get_processing_batch_number()

    def shutdown(self):
        """Shutdown function."""
        return self.__conductor.shutdown()

    def _log_start(self, payloads):
        lu.get_logger().info("Scoring {} requests".format(len(payloads)))
        lu.get_logger().info("ParallelDriver: async mode is {}".format(self._configuration.async_mode))

    def _generate_scoring_requests(self, payloads, mini_batch_context):
        scoring_requests: "list[ScoringRequest]" = []
        failed_scoring_results: "list[ScoringResult]" = []

        for payload in payloads:
            try:
                scoring_request = ScoringRequest(original_payload=payload,
                                                 input_to_request_transformer=self.__input_to_request_transformer,
                                                 input_to_log_transformer=self.__input_to_log_transformer,
                                                 mini_batch_context=mini_batch_context)
                scoring_requests.append(scoring_request)
            except RequestModificationException as e:
                lu.get_logger().error(f"ParallelDriver: RequestModificationException raised: {e}")
                lu.get_logger().info("ParallelDriver: Faking failed ScoringResult, omit=False")
                failed_scoring_results.append(ScoringResult.Failed(scoring_request))

        return scoring_requests, failed_scoring_results
