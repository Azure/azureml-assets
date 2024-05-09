# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for Segmented score context."""

import aiohttp
import json

from .common.common import str2bool
from . import logging_utils as lu
from .scoring_result import ScoringResult, ScoringResultStatus
from .scoring_request import ScoringRequest
from .scoring_client import ScoringClient
from .common.json_encoder_extensions import BatchComponentJSONEncoder


class SegmentedScoreContext:
    """The class for Segmented score context."""

    def __init__(self, original_scoring_request: ScoringRequest, segment_max_token_size: int):
        """Init method."""
        self.__original_scoring_request = original_scoring_request
        self.__supports_segmentation = True
        self.__next_scoring_request: ScoringRequest = None

        payload_object = original_scoring_request.cleaned_payload_obj

        self.__original_max_tokens = None

        if "max_tokens" in payload_object:
            self.__original_max_tokens = int(payload_object.get("max_tokens"))

        if ("n" in payload_object and (int)(payload_object["n"]) > 1) or \
                ("stream" in payload_object and str2bool(payload_object["stream"])) or \
                (self.__original_max_tokens is not None and self.__original_max_tokens <= segment_max_token_size):
            self.__supports_segmentation = False

        self.__segment_max_token_size = segment_max_token_size
        self.__segmented_results: list[ScoringResult] = []
        self.__total_tokens_generated = 0
        self.__cumulative_generated_tokens: str = ""
        self.__last_stop_reason: str = None

    async def score_until_completion(
        self,
        scoring_client: ScoringClient,
        session: aiohttp.ClientSession,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score until completion."""
        next_result: ScoringResult = None

        while self.has_more():
            next_scoring_request = self.__create_next_scoring_request()
            next_result = await scoring_client.score_until_completion(
                session, next_scoring_request, worker_id=worker_id)

            if next_result.status == ScoringResultStatus.SUCCESS:
                self.__add_successful_result(next_result)

        return next_result

    async def score_next_once(
        self,
        scoring_client: ScoringClient,
        session: aiohttp.ClientSession,
        worker_id: str = "1"
    ) -> ScoringResult:
        """Score next once."""
        if self.__next_scoring_request is None:
            self.__next_scoring_request = self.__create_next_scoring_request()

        next_scoring_request = self.__next_scoring_request

        next_result = await scoring_client.score_once(session, next_scoring_request, worker_id=worker_id)

        # Scoring terminated with non-retriable response, reset self.__next_scoring_request
        self.__next_scoring_request = None

        if next_result.status == ScoringResultStatus.SUCCESS:
            self.__add_successful_result(next_result)

        return next_result

    def has_more(self) -> bool:
        """Has more data."""
        if len(self.__segmented_results) > 0:
            if self.__original_max_tokens is None:
                # No provided max_tokens scenario
                return self.__last_stop_reason is None or self.__last_stop_reason != "stop"
            elif self.__supports_segmentation:
                # Incase of max_tokens having a very small value (e.g. 1),
                # it is possible that model cannot return
                # additional tokens. It will return an empty text with finish_reason = "length".
                # We need to treat it as a stop condition.
                if self.__segmented_results[-1].response_body["choices"][0]["text"] == "":
                    lu.get_logger().debug("segmented response generated no additional text. stopping.")
                    return False

                return (self.__last_stop_reason is None or self.__last_stop_reason != "stop") and \
                    self.__total_tokens_generated < self.__original_max_tokens
            else:
                return False

        return True

    def build_scoring_result(self, final_result: ScoringResult):
        """Build scoring result."""
        if not self.__supports_segmentation or len(self.__segmented_results) <= 1:
            return final_result
        else:
            # Fix request object to be the original request object
            final_result.request_obj = self.__original_scoring_request.original_payload_obj

            # Merge output properties
            final_result.response_body["choices"][0]["text"] = self.__cumulative_generated_tokens

            self.__merge_logprobs(final_result)
            self.__merge_usage(final_result)

            final_result.segmented_response_bodies = [x.response_body for x in self.__segmented_results]

            return final_result

    def __merge_logprobs(self, final_result: ScoringResult):
        logprobs_properties = ["tokens", "token_logprobs", "top_logprobs", "text_offset"]

        if "logprobs" in final_result.response_body["choices"][0]:
            final_logprobs = final_result.response_body["choices"][0]["logprobs"]

            if final_logprobs is not None:
                for logprobs_property in logprobs_properties:
                    final_logprobs[logprobs_property] = []

                for current_result in self.__segmented_results:
                    if "logprobs" in current_result.response_body["choices"][0]:
                        current_logprobs = current_result.response_body["choices"][0]["logprobs"]

                        for logprobs_property in logprobs_properties:
                            if logprobs_property in current_logprobs and \
                                    current_logprobs[logprobs_property] is not None:
                                final_logprobs[logprobs_property].extend(current_logprobs[logprobs_property])

    def __merge_usage(self, final_result: ScoringResult):
        usage_properties = ["prompt_tokens", "completion_tokens", "total_tokens"]

        if "usage" in final_result.response_body:
            final_usage = final_result.response_body["usage"]

            if final_usage is not None:
                for usage_property in usage_properties:
                    final_usage[usage_property] = 0

                for current_result in self.__segmented_results:
                    if "usage" in current_result.response_body:
                        current_usage = current_result.response_body["usage"]

                        for usage_property in usage_properties:
                            if usage_property in current_usage:
                                final_usage[usage_property] += int(current_usage[usage_property])

    def __add_successful_result(self, result: ScoringResult):
        self.__segmented_results.append(result)

        response_choice = result.response_body["choices"][0]

        self.__last_stop_reason = response_choice.get("finish_reason", None)
        self.__cumulative_generated_tokens += response_choice["text"]

        if result.completion_tokens is not None:
            self.__total_tokens_generated += result.completion_tokens

        lu.get_logger().debug(
            "Completed segmented request with stop reason {}, generated {} tokens, has more: {}".format(
                self.__last_stop_reason, self.__total_tokens_generated, self.has_more()))

    def __create_next_scoring_request(self) -> ScoringRequest:
        if not self.__supports_segmentation:
            return self.__original_scoring_request

        payload_object = json.loads(self.__original_scoring_request.original_payload)

        if len(self.__segmented_results) > 0:
            payload_object["prompt"] += self.__cumulative_generated_tokens

        max_tokens = self.__segment_max_token_size
        if self.__original_max_tokens is not None:
            max_tokens = min(
                self.__segment_max_token_size, self.__original_max_tokens - self.__total_tokens_generated)

        payload_object["max_tokens"] = max_tokens

        return ScoringRequest(original_payload=json.dumps(payload_object, cls=BatchComponentJSONEncoder))
