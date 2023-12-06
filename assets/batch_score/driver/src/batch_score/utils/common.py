# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import collections.abc
import json
from argparse import ArgumentParser
from urllib.parse import urlparse

import numpy
import pandas as pd

from ..common.scoring.scoring_result import ScoringResult
from . import embeddings_utils as embeddings
from .json_encoder_extensions import BatchComponentJSONEncoder, NumpyArrayEncoder


def get_base_url(url: str) -> str:
    if not url:
        return url
    
    parse_result = urlparse(url)
    return f"{parse_result.scheme}://{parse_result.netloc}"

def backoff(attempt: int, base_delay: float = 1, exponent: float = 2, max_delay: float = 20):
    return min(max_delay, base_delay * attempt**exponent)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError('Boolean value expected.')

def convert_result_list(results: "list[ScoringResult]", batch_size_per_request: int) -> "list[str]":
    output_list: list[dict[str, str]] = []
    for scoringResult in results:
        output: dict[str, str] = {}
        output["status"] = scoringResult.status.name
        output["start"] = scoringResult.start
        output["end"] = scoringResult.end
        output["request"] = scoringResult.request_obj
        output["response"] = scoringResult.response_body

        if scoringResult.segmented_response_bodies != None and len(scoringResult.segmented_response_bodies) > 0:
            output["segmented_responses"] = scoringResult.segmented_response_bodies

        if scoringResult.request_metadata is not None:
            output["request_metadata"] = scoringResult.request_metadata

        if batch_size_per_request > 1:
            batch_output_list = embeddings._convert_to_list_of_output_items(output, scoringResult.estimated_token_counts)
            output_list.extend(batch_output_list)
        else:
            output_list.append(output)

    return list(map(__stringify_output, output_list))

def convert_to_list(data: pd.DataFrame, additional_properties:str = None, batch_size_per_request:int = 1) -> "list[str]":
    columns = data.keys()
    payloads = []
    additional_properties_list = None
    
    # Per https://platform.openai.com/docs/api-reference/
    int_forceable_properties = ["max_tokens", "n", "logprobs", "best_of", "n_epochs", "batch_size", "classification_n_classes"]

    if additional_properties is not None:
        additional_properties_list = json.loads(additional_properties)

    if batch_size_per_request > 1:
        batched_payloads = embeddings._convert_to_list_of_input_batches(data, batch_size_per_request)
        payloads.extend(batched_payloads)
    else:
        for row in data.values.tolist():
            payload_obj: dict[str, any] = {}
            for indx, col in enumerate(columns):
                payload_val = row[indx]
                if isinstance(payload_val, collections.abc.Sequence) or isinstance(payload_val, numpy.ndarray) or\
                    not pd.isnull(payload_val):
                    if col in int_forceable_properties:
                        payload_val = int(payload_val)
                    payload_obj[col] = payload_val
            payloads.append(payload_obj)

    for idx, payload in enumerate(payloads):
        # Payloads are mutable. Update them with additional properties and put them back in the list as strings.
        __add_properties_to_payload_object(payload, additional_properties_list)
        payloads[idx] = __stringify_payload(payload)

    return payloads

def __add_properties_to_payload_object(payload_obj: dict,
                                       additional_properties_list: dict):
    if additional_properties_list is not None:
        for key, value in additional_properties_list.items():
            payload_obj[key] = value

def __stringify_output(payload_obj: dict) -> str:
    return json.dumps(payload_obj, cls=BatchComponentJSONEncoder)

def __stringify_payload(payload_obj: dict) -> str:
    return json.dumps(payload_obj, cls=NumpyArrayEncoder)

