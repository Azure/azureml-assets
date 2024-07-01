# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utilities."""

<<<<<<< HEAD
import json
from argparse import ArgumentParser
from urllib.parse import urlparse

from ..common.scoring.scoring_result import ScoringResult
from . import embeddings_utils as embeddings
from .json_encoder_extensions import BatchComponentJSONEncoder

=======
from argparse import ArgumentParser
from urllib.parse import urlparse

>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa

def get_base_url(url: str) -> str:
    """Get base url."""
    if not url:
        return url

    parse_result = urlparse(url)
    return f"{parse_result.scheme}://{parse_result.netloc}"


def backoff(attempt: int, base_delay: float = 1, exponent: float = 2, max_delay: float = 20):
    """Calculate backoff delay time."""
    return min(max_delay, base_delay * attempt**exponent)


def str2bool(v):
    """Convert string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError('Boolean value expected.')


<<<<<<< HEAD
def convert_result_list(results: "list[ScoringResult]", batch_size_per_request: int) -> "list[str]":
    """Convert scoring results to the result list."""
    output_list: list[dict[str, str]] = []
    for scoringResult in results:
        output: dict[str, str] = {}
        output["status"] = scoringResult.status.name
        output["start"] = scoringResult.start
        output["end"] = scoringResult.end
        output["request"] = scoringResult.request_obj
        output["response"] = scoringResult.response_body

        if scoringResult.segmented_response_bodies is not None and len(scoringResult.segmented_response_bodies) > 0:
            output["segmented_responses"] = scoringResult.segmented_response_bodies

        if scoringResult.request_metadata is not None:
            output["request_metadata"] = scoringResult.request_metadata

        if batch_size_per_request > 1:
            batch_output_list = embeddings._convert_to_list_of_output_items(
                output,
                scoringResult.estimated_token_counts)
            output_list.extend(batch_output_list)
        else:
            output_list.append(output)

    return list(map(__stringify_output, output_list))


=======
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
def get_mini_batch_id(mini_batch_context: any):
    """Get mini batch id from mini batch context."""
    if mini_batch_context:
        return mini_batch_context.mini_batch_id
<<<<<<< HEAD


def __stringify_output(payload_obj: dict) -> str:
    return json.dumps(payload_obj, cls=BatchComponentJSONEncoder)
=======
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
