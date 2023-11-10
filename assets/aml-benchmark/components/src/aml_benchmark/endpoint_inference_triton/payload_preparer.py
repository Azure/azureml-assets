# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Payload Preparer File."""

from typing import List, Any, Tuple, Dict
import json

import numpy as np
import gevent.ssl
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype

from aml_benchmark.utils.constants import TaskType


def _generate_inputs_triton(task_type: str, **kwargs: Any) -> List[httpclient.InferInput]:
    """Create the input for the triton inference server."""

    def _prepare_tensor(name: str, input: np.array) -> httpclient.InferInput:
        infer_input = httpclient.InferInput(name, input.shape, np_to_triton_dtype(input.dtype))
        infer_input.set_data_from_numpy(input)
        return infer_input
    
    def _update_dict_with_default_params(dict_to_update: dict, default_params: dict) -> None:
        for key, value in default_params.items():
            if dict_to_update.get(key, None) is None:
                dict_to_update[key] = value

    default_params = {
        "top_k": 1,
        "top_p": 0,
        "temperature": 1.0,
    }
    _update_dict_with_default_params(kwargs, default_params)

    top_p_arr = np.array([kwargs["top_p"]]).astype(np.float32).reshape((1, -1))
    temperature_arr = np.array([kwargs["temperature"]]).astype(np.float32).reshape((1, -1))

    inputs = [
        _prepare_tensor("top_p", top_p_arr),
        _prepare_tensor("temperature", temperature_arr),
    ]

    if task_type == TaskType.TEXT_GENERATION.value:
        default_params["max_output_token"] = 100
        _update_dict_with_default_params(kwargs, default_params)

        top_k_arr = np.array([kwargs["top_k"]]).astype(np.int64).reshape((1, -1))
        prompts_arr = np.array([kwargs["prompt"]]).astype(object).reshape((1, -1))
        max_output_token_arr = np.array([kwargs["max_output_token"]]).astype(np.int64).reshape((1, -1))

        inputs.extend([
            _prepare_tensor("top_k", top_k_arr),
            _prepare_tensor("prompts", prompts_arr),
            _prepare_tensor("max_output_token", max_output_token_arr),
        ])

    elif task_type == TaskType.CHAT_COMPLETION.value:
        default_params["max_tokens"] = 300
        default_params["length_penalty"] = 1.0
        default_params["repetition_penalty"] = 1.0
        default_params["random_seed"] = 0
        default_params["beam_width"] = 1
        default_params["stream"] = False
        _update_dict_with_default_params(kwargs, default_params)

        GPT_PROMPT_TEMPLATE = ( 
            "<extra_id_0>System\n"
            "A chat between a curious user and an artificial intelligence assistant." 
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            "<extra_id_1>User\n"
            "{prompt}\n"
            "<extra_id_1>Assistant\n"
        )
        top_k_arr = np.array([kwargs["top_k"]]).astype(np.uint32).reshape((1, -1))
        _text_input = GPT_PROMPT_TEMPLATE.format(prompt=kwargs["text_input"])
        text_input_arr = prompts_arr = np.array([_text_input]).astype(object).reshape((1, -1))
        max_tokens_arr = np.array([kwargs["max_tokens"]]).astype(np.uint32).reshape((1, -1))
        length_penalty_arr = np.array([kwargs["length_penalty"]]).astype(np.float32).reshape((1, -1))
        repetition_penalty_arr = np.array([kwargs["repetition_penalty"]]).astype(np.float32).reshape((1, -1))
        random_seed_arr = np.array([kwargs["random_seed"]]).astype(np.uint64).reshape((1, -1))
        beam_width_arr = np.array([kwargs["beam_width"]]).astype(np.uint32).reshape((1, -1))
        stream_arr = np.array([kwargs["stream"]], dtype=bool).reshape((1, -1))

        inputs.extend([
            _prepare_tensor("top_k", top_k_arr),
            _prepare_tensor("text_input", text_input_arr),
            _prepare_tensor("max_tokens", max_tokens_arr),
            _prepare_tensor("length_penalty", length_penalty_arr),
            _prepare_tensor("repetition_penalty", repetition_penalty_arr),
            _prepare_tensor("random_seed", random_seed_arr),
            _prepare_tensor("beam_width", beam_width_arr),
            _prepare_tensor("stream", stream_arr),
        ])

    return inputs


def prepare_payload(data: List[Dict[str, Any]], task_type: str) -> List[List[httpclient.InferInput]]:
    """Prepare the payload."""
    payloads: List[List[httpclient.InferInput]] = []

    for row in data:
        inputs = _generate_inputs_triton(task_type, **row)
        payloads.append(inputs)

    return payloads