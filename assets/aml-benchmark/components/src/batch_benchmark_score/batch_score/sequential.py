# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The sequential driver for endpoint input preparer."""

import asyncio
import aiohttp
import random
import os

from aiohttp import TraceConfig
from .utils.common.common import convert_result_list
from .utils import logging_utils as lu
from .utils.logging_utils import get_events_client
from .utils.scoring_result import ScoringResult, ScoringResultStatus
from .utils.scoring_request import ScoringRequest
from .utils.scoring_client import ScoringClient
from .utils.segmented_score_context import SegmentedScoreContext
from .request_modification.input_transformer import InputTransformer
from .request_modification.modifiers.request_modifier import RequestModificationException


class Sequential:
    """Sequential endpoint class."""

    def __init__(self,
                 scoring_client: ScoringClient,
                 segment_large_requests: str,
                 segment_max_token_size: int,
                 trace_configs: "list[TraceConfig]" = None,
                 request_input_transformer: InputTransformer = None,
                 logging_input_transformer: InputTransformer = None) -> None:
        """Init call for sequential endpoint call."""
        self.args = None
        self.id = "-S-" + str(random.randint(0, 9999))

        self.__scoring_client: ScoringClient = scoring_client
        self.__segment_large_requests: str = segment_large_requests
        self.__segment_max_token_size: int = segment_max_token_size

        self.__trace_configs = trace_configs
        self.__request_input_transformer = request_input_transformer
        self.__logging_input_transformer = logging_input_transformer

    async def main(self, scoring_requests: "list[ScoringRequest]"):
        """Sequential entry."""
        result: list[ScoringResult] = []
        total = 30 * 60  # Default timeout to 30 minutes
        if self.__segment_large_requests == "enabled":
            total = self.__segment_max_token_size * 1  # Assume 1 second per token generation
            total = max(total, 3 * 60)  # Lower bound of 3 minutes

        timeout_override = os.environ.get("BATCH_SCORE_REQUEST_TIMEOUT")
        total = int(timeout_override) if timeout_override else total

        lu.get_logger().info(f"Setting client-side request timeout to {total} seconds.")
        timeout = aiohttp.ClientTimeout(total=total)

        async with aiohttp.ClientSession(
                timeout=timeout,
                trace_configs=self.__trace_configs
        ) as session:
            for indx, scoring_request in enumerate(scoring_requests):
                scoring_result: ScoringResult = None

                if self.__segment_large_requests == "enabled":
                    segmented_score_context = SegmentedScoreContext(
                        scoring_request, self.__segment_max_token_size)
                    scoring_result = await segmented_score_context.score_until_completion(
                        scoring_client=self.__scoring_client,
                        session=session
                    )
                    scoring_result = segmented_score_context.build_scoring_result(
                        final_result=scoring_result)
                else:
                    scoring_result = await self.__scoring_client.score_until_completion(
                        session=session,
                        scoring_request=scoring_request
                    )

                lu.get_logger().debug(
                    "{}: Finished scoring of scoring_request #{}".format(self.id, indx))

                get_events_client().emit_row_completed(1, str(scoring_result.status))

                if not scoring_result.omit:
                    result.append(scoring_result)

        return result

    def start(self, payloads: "list[str]"):
        """Start the endpoint call."""
        lu.get_logger().info("{}: Scoring {} lines".format(self.id, len(payloads)))
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
                print(e)
                lu.get_logger().info(
                    "RequestModificationException raised. Faking failed ScoringResult, omit=False")
                scoring_results.append(
                    ScoringResult(
                        status=ScoringResultStatus.FAILURE, omit=False,
                        start=0, end=0, request_obj=None, request_metadata=None,
                        response_body=None, response_headers=None, num_retries=0))

        scoring_results.extend(asyncio.run(self.main(scoring_requests)))

        if self.__logging_input_transformer:
            for scoring_result in scoring_results:
                scoring_result.request_obj = self.__logging_input_transformer.apply_modifications(
                    request_obj=scoring_result.request_obj)

        results = convert_result_list(results=scoring_results)

        return results
