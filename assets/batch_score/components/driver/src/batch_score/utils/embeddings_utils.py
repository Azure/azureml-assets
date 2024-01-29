# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Embeddings utilities."""

from copy import deepcopy

import pandas as pd

from ..batch_pool.quota.estimators import EmbeddingsEstimator
from ..common.telemetry import logging_utils as lu

estimator = None


def _convert_to_list_of_input_batches(
        data: pd.DataFrame,
        batch_size_per_request: int) -> "list[dict]":
    """Convert input data to a list of input batches."""
    """This method is specific for APIs that allow batching, currently only Embeddings.
    That means the data has the "input" column.

    Given a dataframe and batch size, convert the data into a list of dictionaries,
    where each element has an "input" list of strings equal* to the batch size.
    *The last element's "input" list of strings will have length in [1, batch_size_per_request].
    """
    numrows = len(data)
    list_of_input_batches = []

    for i in range(0, numrows, batch_size_per_request):
        list_of_strings = data["input"][i: i + batch_size_per_request].values.tolist()
        payload_obj = {"input": list_of_strings}
        list_of_input_batches.append(payload_obj)
    return list_of_input_batches


def _convert_to_list_of_output_items(result: dict, token_count_estimates: "tuple[int]") -> "list[dict]":
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
    response_obj = result["response"]
    try:
        response_data = response_obj.pop('data', None)
    except AttributeError:
        response_data = None
    request = result["request"]
    numrequests = len(request["input"])

    if response_data is not None:
        # Result has data; response_obj["data"]
        numresults = len(response_data)
        __validate_response_data_length(numrequests, numresults)
        output_index_to_embedding_info_map = __build_output_idx_to_embedding_mapping(response_data)
        update_prompt_tokens = __tiktoken_estimates_succeeded(token_count_estimates, numrequests)
        if not update_prompt_tokens:
            # Single online endpoints will not have computed token estimates, as this occurs in quota client.
            token_count_estimates = __tiktoken_estimates_retry(request)
            update_prompt_tokens = __tiktoken_estimates_succeeded(token_count_estimates, numrequests)
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
        single_output = {"request": deepcopy(request), "response": deepcopy(response_obj)}
        single_output["request"]["input"] = input_batch[i]
        if numresults > -1:
            single_output["response"]["data"] = [output_index_to_embedding_info_map[i]]
            if update_prompt_tokens:
                __override_prompt_tokens(single_output, token_count_estimates[i])
        output_list.append(single_output)

    return output_list


def __build_output_idx_to_embedding_mapping(response_data):
    """Build a mapping from output index to embedding."""
    """
    Given response data, return a dictionary of the index and embedding info for each element of the batch.
    Unsure if the responses are always in the correct order by input index, ensure output order by mapping out index.

    Args:
        response_data: The list of outputs from the 'data' of API response.

    Returns:
        Dict mapping index to embedding info.
    """
    return {embedding_info['index']: embedding_info for embedding_info in response_data}


def __override_prompt_tokens(output_obj, token_count):
    """
    Set the token_count as the value for `prompt_tokens` in response's usage info.

    Args:
        output_obj: The dictionary of info for response, request
        token_count: The tiktoken count for this input string
    """
    try:
        output_obj["response"]["usage"]["prompt_tokens"] = token_count
    except Exception as exc:
        lu.get_logger().exception("Unable to set prompt token override.")
        raise exc


def __tiktoken_estimates_succeeded(token_count_estimates: "tuple[int]", input_length: int) -> bool:
    """
    Return True if the length of the batch of inputs matches the length of the tiktoken estimates.

    Args:
        token_count_estimates: The tuple of tiktoken estimates for the inputs in this batch
        input_length: The length of inputs in this batch
    """
    token_est_length = len(token_count_estimates)
    length_matches = token_est_length == input_length
    if not length_matches:
        lu.get_logger().warn(f"Input length {input_length} does not match token estimate length {token_est_length}. "
                             "Skipping prompt_tokens count overrides.")
    return length_matches


def __tiktoken_estimates_retry(request_obj: dict) -> "tuple[int]":
    """
    Return token counts for the inputs within a batch.

    Args:
        request_obj: The request dictionary.
    """
    lu.get_logger().debug("Attempting to calculate tokens for the embedding input batch.")
    global estimator
    if estimator is None:
        estimator = EmbeddingsEstimator()
    token_counts = estimator.estimate_request_cost(request_obj)
    if token_counts == 1:
        # This occurs if tiktoken module fails. See DV3Estimator for more info on why this could fail.
        return ()
    else:
        return token_counts


def __validate_response_data_length(numrequests, numresults):
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
