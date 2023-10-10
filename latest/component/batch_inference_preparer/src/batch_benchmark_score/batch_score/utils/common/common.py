# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Common utitility methods."""

import json
import pandas as pd
import numpy
import collections.abc

from urllib.parse import urlparse
from ..scoring_result import ScoringResult, ScoringResultStatus
from .json_encoder_extensions import BatchComponentJSONEncoder, NumpyArrayEncoder

from azureml._common._error_definition.azureml_error import AzureMLError
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


def get_base_url(url: str) -> str:
    """Get base url."""
    if not url:
        return url

    parse_result = urlparse(url)
    return f"{parse_result.scheme}://{parse_result.netloc}"


def backoff(attempt: int, base_delay: float = 1, exponent: float = 2, max_delay: float = 20):
    """Get backoff time."""
    return min(max_delay, base_delay * attempt**exponent)


def str2bool(v):
    """Convert str to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details='Boolean value expected.')
        )


def convert_result_list(results: "list[ScoringResult]") -> "list[str]":
    """Convert result list."""
    output_list: list[dict[str, str]] = []
    for scoringResult in results:
        output: dict[str, str] = {}
        output["status"] = "success" if (scoringResult.status == ScoringResultStatus.SUCCESS) else "failure"
        output["request"] = scoringResult.request_obj
        output["response"] = scoringResult.response_body
        output["start"] = scoringResult.start * 1000
        output["end"] = scoringResult.end * 1000
        output["latency"] = (scoringResult.end - scoringResult.start) * 1000

        if scoringResult.segmented_response_bodies is not None and len(scoringResult.segmented_response_bodies) > 0:
            output["segmented_responses"] = scoringResult.segmented_response_bodies

        if scoringResult.request_metadata is not None:
            output["request_metadata"] = scoringResult.request_metadata

        output_list.append(json.dumps(output, cls=BatchComponentJSONEncoder))

    return output_list


def convert_to_list(data: pd.DataFrame, additional_properties: str = None) -> "list[str]":
    """Convert data to list."""
    columns = data.keys()
    payloads: list[str] = []
    additional_properties_list = None

    # Per https://platform.openai.com/docs/api-reference/
    int_forceable_properties = [
        "max_tokens", "n", "logprobs", "best_of", "n_epochs", "batch_size", "classification_n_classes"]

    if additional_properties is not None:
        additional_properties_list = json.loads(additional_properties)

    for row in data.values.tolist():
        payload_obj: dict[str, any] = {}
        for indx, col in enumerate(columns):
            payload_val = row[indx]
            if isinstance(payload_val, collections.abc.Sequence) or isinstance(payload_val, numpy.ndarray) or\
                    not pd.isnull(payload_val):
                if col in int_forceable_properties:
                    payload_val = int(payload_val)
                payload_obj[col] = payload_val

        if additional_properties_list is not None:
            for key, value in additional_properties_list.items():
                payload_obj[key] = value

        payload = json.dumps(payload_obj, cls=NumpyArrayEncoder)
        payloads.append(payload)

    return payloads
