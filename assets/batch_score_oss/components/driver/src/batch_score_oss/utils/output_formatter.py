# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for the output formatter."""

import json
from abc import ABC, abstractmethod

from ..common.scoring.scoring_result import ScoringResult
from .json_encoder_extensions import BatchComponentJSONEncoder
from ..common.telemetry import logging_utils as lu


class OutputFormatter(ABC):
    """An abstract class for formatting output."""

    @abstractmethod
    def format_output(self, results: "list[ScoringResult]", batch_size_per_request: int) -> "list[str]":
        """Abstract output formatting method."""
        pass

    @abstractmethod
    def _get_response_obj(self, result: dict):
        pass

    @abstractmethod
    def _get_custom_id(self, result: dict):
        pass

    @abstractmethod
    def _get_single_output(self, *args):
        pass

    def _convert_to_list_of_output_items(self, result: dict, token_count_estimates: "tuple[int]") -> "list[dict]":
        """Convert results to a list of output items."""
        """
        Only used by the Embeddings API with batched HTTP requests, this method takes
        a scoring result as dictionary of "request", "response", and (optional) "request_metadata".
        It returns the batch list within request and response mapped out into a list of dictionaries,
        each with the correlating request and response from the batch.
        If the scoring result has request metadata, this is persisted in each of the
        output dictionaries.

        Args:
            result: The scoring result containing a batch of inputs and outputs.
            token_count_estimates: The tuple of tiktoken estimates for each input in the batch.

        Returns:
            List of output objects, each with "request", "response", (optional) "request_metadata".
        """
        output_list = []

        response_obj = self._get_response_obj(result)
        custom_id = self._get_custom_id(result)
        format_version = 1 if custom_id is None else 2

        try:
            response_data = response_obj.pop('data', None)
        except AttributeError:
            response_data = None
        request = result["request"]
        numrequests = len(request["input"])

        if response_data is not None:
            # Result has data; response_obj["data"]
            numresults = len(response_data)
            self.__validate_response_data_length(numrequests, numresults)
            output_index_to_embedding_info_map = self.__build_output_idx_to_embedding_mapping(response_data)
            update_prompt_tokens = self.__tiktoken_estimates_succeeded(token_count_estimates, numrequests)
        else:
            # Result has error; response_obj["error"]. Copy this for each request in batch below.
            numresults = -1
            error_message = "The batch request resulted in an error. See job output for the error message."
            lu.get_logger().error(error_message)

        # Input can be large. Pop and iterate through the batch to avoid copying repeatedly.
        input_batch = request.pop('input', None)

        for i in range(numrequests):
            # The large "input" from request and "data" from response have been popped so copy is smaller.
            # "input" and "data" are set for each below.
            if format_version == 1:
                single_output = self._get_single_output(request, response_obj, input_batch[i])
            else:
                single_output = self._get_single_output(custom_id, result)

            if numresults > -1:
                if format_version == 1:
                    single_output["response"]["data"] = [output_index_to_embedding_info_map[i]]
                else:
                    single_output["response"]["body"]["data"] = [output_index_to_embedding_info_map[i]]
                if update_prompt_tokens:
                    self.__override_prompt_tokens(single_output, token_count_estimates[i], format_version)
            output_list.append(single_output)

        return output_list

    def __build_output_idx_to_embedding_mapping(self, response_data):
        """Build a mapping from output index to embedding."""
        """
        Given response data, return a dictionary of the index and embedding info for each element of the batch.
        Unsure if the responses are always in the correct order by input index,
        ensure output order by mapping out index.

        Args:
            response_data: The list of outputs from the 'data' of API response.

        Returns:
            Dict mapping index to embedding info.
        """
        return {embedding_info['index']: embedding_info for embedding_info in response_data}

    def __override_prompt_tokens(self, output_obj, token_count, format_version):
        """
        Set the token_count as the value for `prompt_tokens` in response's usage info.

        Args:
            output_obj: The dictionary of info for response, request
            token_count: The tiktoken count for this input string
            format_version: The output format version
        """
        try:
            if format_version == 1:
                output_obj["response"]["usage"]["prompt_tokens"] = token_count
            else:
                output_obj["response"]["body"]["usage"]["prompt_tokens"] = token_count
        except Exception as exc:
            lu.get_logger().exception("Unable to set prompt token override.")
            raise exc

    def __tiktoken_estimates_succeeded(self, token_count_estimates: "tuple[int]", input_length: int) -> bool:
        """
        Return True if the length of the batch of inputs matches the length of the tiktoken estimates.

        Args:
            token_count_estimates: The tuple of tiktoken estimates for the inputs in this batch
            input_length: The length of inputs in this batch
        """
        token_est_length = len(token_count_estimates)
        length_matches = token_est_length == input_length
        if not length_matches:
            lu.get_logger().warn(f"Input length {input_length} does not match token estimate "
                                 "length {token_est_length}. Skipping prompt_tokens count overrides.")
        return length_matches

    def __validate_response_data_length(self, numrequests, numresults):
        """
        Validate the number of outputs from the API response matches the number of requests in the batch.

        Args:
            numrequests: The number of requests in this batch.
            numresults: The number of results in the response data.

        Raises:
            Exception if response length and request length do not match.
        """
        if numresults != numrequests:
            error_message = f"Result data length {numresults} != " + \
                            f"{numrequests} request batch length."
            lu.get_logger().error(error_message)
            raise Exception(error_message)

    def _stringify_output(self, payload_obj: dict) -> str:
        return json.dumps(payload_obj, cls=BatchComponentJSONEncoder)
