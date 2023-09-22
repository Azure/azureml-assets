# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The parallel driver for endpoint input preparer."""

import asyncio

from .utils.common.common import convert_result_list
from .utils import logging_utils as lu
from .utils.scoring_request import ScoringRequest
from .utils.scoring_result import ScoringResult, ScoringResultStatus
from .parallel.conductor import Conductor
from .request_modification.input_transformer import InputTransformer
from .request_modification.modifiers.request_modifier import RequestModificationException


class Parallel:
    """Parallel driver class."""

    def __init__(self,
                 loop: asyncio.AbstractEventLoop,
                 conductor: Conductor,
                 request_input_transformer: InputTransformer = None,
                 logging_input_transformer: InputTransformer = None):
        """Init for parallel driver."""
        self.__loop = loop
        self.__conductor: Conductor = conductor
        self.__request_input_transformer: InputTransformer = request_input_transformer
        self.__logging_input_transformer: InputTransformer = logging_input_transformer

    def close_session(self):
        """Close the parallel session."""
        self.__conductor.close_session()

    def start(self, payloads: "list[str]"):
        """Start parallel call."""
        lu.get_logger().info("Scoring {} lines".format(len(payloads)))
        scoring_requests: "list[ScoringRequest]" = []
        scoring_results: "list[ScoringResult]" = []

        for payload in payloads:
            try:
                scoring_request = ScoringRequest(
                    original_payload=payload,
                    request_input_transformer=self.__request_input_transformer,
                    logging_input_transformer=self.__logging_input_transformer)
                scoring_requests.append(scoring_request)
            except RequestModificationException as e:
                lu.get_logger().info(f"RequestModificationException raised: {e}")
                lu.get_logger().info("Faking failed ScoringResult, omit=False")
                scoring_results.append(
                    ScoringResult(
                        status=ScoringResultStatus.FAILURE, omit=False,
                        start=0, end=0, request_obj=None, request_metadata=None,
                        response_body=None, response_headers=None, num_retries=0))

        scoring_results.extend(self.__loop.run_until_complete(self.__conductor.run(scoring_requests)))

        if self.__logging_input_transformer:
            for scoring_result in scoring_results:
                # None request_obj values may be present,
                # if a RequestModificationException was thrown during ScoringRequest creation
                if not scoring_result.omit and scoring_result.request_obj:
                    scoring_result.request_obj = self.__logging_input_transformer.apply_modifications(
                        request_obj=scoring_result.request_obj)

        results = convert_result_list(results=scoring_results)

        return results
