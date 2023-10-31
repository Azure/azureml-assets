# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""VLLM Engine module.

This module contains the VLLMEngine class which is responsible for initializing the VLLM server,
generating responses for given prompts, and managing the server processes.
"""

# flake8: noqa

import json
import os
import socket
import subprocess
import time
import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import requests
import torch.cuda

from configs import EngineConfig, TaskConfig
from engine.engine import AbstractEngine, InferenceResult
from logging_config import configure_logger
from utils import log_execution_time

logger = configure_logger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


# fmt: off
VLLM_SAMPLING_PARAMS = {
    "n": "Number of output sequences to return for the given prompt.",
    "best_of": "Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`.",
    "presence_penalty": "Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
    "frequency_penalty": "Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.",
    "temperature": "Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.",
    "top_p": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. Set to 1 to consider all tokens.",
    "top_k": "Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens.",
    "use_beam_search": "Whether to use beam search instead of sampling.",
    "length_penalty": "Float that penalizes sequences based on their length. Used in beam search.",
    "early_stopping": 'Controls the stopping condition for beam search. It accepts the following values: `True`, where the generation stops as soon as there are `best_of` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).',
    "stop": "List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.",
    "stop_token_ids": "List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens.",
    "ignore_eos": "Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.",
    "max_tokens": "Maximum number of tokens to generate per output sequence.",
    "logprobs": "Number of log probabilities to return per output token.",
    "skip_special_tokens": "Whether to skip special tokens in the output. Defaults to true.",
    "_batch_size": "Number of prompts to generate in parallel. Defaults to 1.",
}


# fmt: on


class VLLMEngine(AbstractEngine):
    """VLLM Engine class.

    This class is responsible for initializing the VLLM server, generating responses for given prompts,
    and managing the server processes.
    """

    def __init__(self, engine_config: EngineConfig, task_config: TaskConfig):
        """Initialize the VLLMEngine with the given engine and task configurations."""
        self.engine_config: EngineConfig = engine_config
        self.task_config: TaskConfig = task_config
        self._server_process: Optional[subprocess.Popen] = None
        self._vllm_args: Dict = self.engine_config.vllm_config or {}

        # Not all vllm arguments require a value
        self._vllm_additional_args: List[str] = []
        self._verify_and_convert_float_type()
        self._load_vllm_defaults()

    @log_execution_time
    def load_model(self):
        """Load the model from the pretrained model specified in the engine configuration."""
        self._start_server(self._vllm_args, self._vllm_additional_args)

    def init_client(self):
        """Initialize client[s] for the engine to receive requests on."""
        self.wait_until_server_healthy(self._vllm_args["host"], self._vllm_args["port"])

    def _load_vllm_defaults(self):
        """Load default values for VLLM server arguments, if not provided."""
        if "host" not in self._vllm_args:
            self._vllm_args["host"] = DEFAULT_HOST
        if "port" not in self._vllm_args:
            self._vllm_args["port"] = DEFAULT_PORT
        if "tensor-parallel-size" not in self._vllm_args:
            self._vllm_args["tensor-parallel-size"] = (
                self.engine_config.tensor_parallel
                if self.engine_config.tensor_parallel is not None
                else torch.cuda.device_count()
            )
        if "model" not in self._vllm_args:
            self._vllm_args["model"] = self.engine_config.model_id
        if "model_name" in self._vllm_args:
            if self._vllm_args["model_name"] == "falcon":
                self._vllm_args["gpu-memory-utilization"] = 1
                self._vllm_additional_args.append("trust-remote-code")
            del self._vllm_args["model_name"]
    
    def _verify_and_convert_float_type(self):
        """
        Check to see whether the model's float type is compatible with the compute type selected.

        Bfloat16 is only supported on GPUs such as A100s. V100s do not support bfloat16, only float16.
        Converting from bfloat16 to float16 is ok in this case.
        """
        # Check if the GPU supports the dtype.
        dtype = self._vllm_args.get("dtype", "auto")
        if dtype == "bfloat16":
            compute_capability = torch.cuda.get_device_capability()
            if compute_capability[0] < 8:
                gpu_name = torch.cuda.get_device_name()

                # Cast bfloat16 to float16
                self._vllm_args["dtype"] = "float16"
                logger.warning(
                    "Bfloat16 is only supported on GPUs with compute capability "
                    f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                    f"{compute_capability[0]}.{compute_capability[1]}. "
                    f"Bfloat16 will be converted to float16.")

    def _start_server(self, server_kwargs: Dict, server_args: List):
        """Start the VLLM server with the given arguments."""
        cmd = ["python", "-m", "vllm.entrypoints.api_server"]
        cmd.extend([f"--{k}={v}" for k, v in server_kwargs.items()])
        cmd.extend([f"--{arg}" for arg in server_args])
        logger.info(f"Starting VLLM server with command: {cmd}")
        self._server_process = subprocess.Popen(cmd)
        logger.info("Starting VLLM server...")

    def _stop_server(self):
        """Stop the VLLM server."""
        if self._server_process:
            self._server_process.terminate()
            logger.info("VLLM server stopped.")

    def _get_generate_uri(self) -> str:
        """Get the URI for the generate endpoint of the VLLM server."""
        return f"http://{self._vllm_args['host']}:{self._vllm_args['port']}/generate"

    def __del__(self):
        """Stop the VLLM server when the VLLMEngine object is deleted."""
        self._stop_server()

    def _gen_params_to_vllm_params(self, params: Dict) -> Dict:
        """Convert generation parameters to VLLM parameters."""
        if "max_gen_len" in params:
            params["max_tokens"] = params["max_gen_len"]

        if "max_new_tokens" in params:
            params["max_tokens"] = params["max_new_tokens"]

        if "do_sample" in params and not params["do_sample"]:
            logger.info("do_sample is false, setting temperature to 0.")
            params["temperature"] = 0.0

        if "use_beam_search" in params and params["use_beam_search"]:
            logger.info("Beam search is enabled, setting temperature to 0.")
            params["temperature"] = 0.0

            if "best_of" not in params:
                logger.info("Beam search is enabled, setting best_of to 2.")
                params["best_of"] = 2

        # Remove unsupported keys and log a warning for each
        unsupported_keys = set(params.keys()) - set(VLLM_SAMPLING_PARAMS.keys())
        for key in unsupported_keys:
            logger.warning(
                f"Warning: Parameter '{key}' is not supported by VLLM and will be removed."
            )
            del params[key]

        return params

    @log_execution_time
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Generate responses for the given prompts with the given parameters."""
        # pop _batch_size from params if it exists, set it to 1 by default (for testing only)
        batch_size = params.pop("_batch_size", 1)

        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                executor.map(
                    lambda prompt: self._generate_single_prompt(prompt, params), prompts
                )
            )

        return results

    @log_execution_time
    def _generate_single_prompt(self, prompt: str, params: Dict) -> InferenceResult:
        """Generate a response for a single prompt with the given parameters."""
        api_url = self._get_generate_uri()
        unfiltered_params = copy.deepcopy(params)
        params = self._gen_params_to_vllm_params(params)
        headers = {"User-Agent": "VLLMEngine Client"}

        payload = {
            "prompt": prompt,
            **params,
            "stream": False,
        }
        start_time = time.time()
        response = requests.post(api_url, headers=headers, json=payload)
        end_time = time.time()
        if response.status_code == 200:
            # take the first candidate from the list of candidates (if beam search was used)
            output = json.loads(response.content)["text"][0]
            generated_text = self._del_prompt_if_req(prompt, output, unfiltered_params)
            inference_time_ms = (end_time - start_time) * 1000
            logger.info(f"Inference time: {inference_time_ms} ms")

            # TODO: Until mii returns the num tokens, approximate num_tokens. roughly, 75 words ~= 100 tokens
            num_tokens = (
                len(self._del_prompt_if_req(prompt, output, unfiltered_params, force=True).split(" "))
                // 75
                * 100
            )
            time_per_token_ms = inference_time_ms / num_tokens if num_tokens > 0 else 0
            logger.info(
                f"Num tokens: {num_tokens}, Time per token: {time_per_token_ms} ms"
            )
            res = InferenceResult(generated_text, inference_time_ms, time_per_token_ms)
        else:
            res = InferenceResult(None, None, None, error=response.content)

        return res
