# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import concurrent
import json
import subprocess
import time
import logging
from concurrent.futures import ThreadPoolExecutor

import requests
import socket
from typing import Dict, List, Optional

from configs import EngineConfig, TaskConfig
from constants import TaskType
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
    def __init__(self, engine_config: EngineConfig, task_config: TaskConfig):
        self.engine_config: EngineConfig = engine_config
        self.task_config: TaskConfig = task_config
        self._server_process: Optional[subprocess.Popen] = None
        self._vllm_args: Dict = self.engine_config.vllm_config or {}
        self._load_vllm_defaults()

    @log_execution_time
    def load_model(self):
        server_args = self._load_vllm_defaults()
        self._start_server(server_args)
        self._wait_until_server_healthy()

    def _load_vllm_defaults(self):
        """Load default values for VLLM server arguments, if not provided."""
        if "host" not in self._vllm_args:
            self._vllm_args["host"] = DEFAULT_HOST
        if "port" not in self._vllm_args:
            self._vllm_args["port"] = DEFAULT_PORT
        if "tensor-parallel-size" not in self._vllm_args:
            self._vllm_args["tensor-parallel-size"] = self.engine_config.tensor_parallel
        if "model" not in self._vllm_args:
            self._vllm_args["model"] = self.engine_config.model_id

        return self._vllm_args

    def _start_server(self, server_args: Dict):
        cmd = ["python", "-m", "vllm.entrypoints.api_server"]
        cmd.extend([f"--{k}={v}" for k, v in server_args.items()])
        logger.info(f"Starting VLLM server with command: {cmd}")
        self._server_process = subprocess.Popen(cmd)
        logger.info("Starting VLLM server...")

    @log_execution_time
    def _wait_until_server_healthy(self):
        start_time = time.time()
        while time.time() - start_time < 15 * 60:  # 15 minutes
            is_healthy = is_port_open(self._vllm_args["host"], self._vllm_args["port"])
            if is_healthy:
                logger.info("VLLM server is healthy.")
                return
            logger.info("Waiting for VLLM server to start...")
            time.sleep(30)
        raise Exception("VLLM server did not become healthy within 15 minutes.")

    def _stop_server(self):
        if self._server_process:
            self._server_process.terminate()
            logger.info("VLLM server stopped.")

    def _get_generate_uri(self) -> str:
        return f"http://{self._vllm_args['host']}:{self._vllm_args['port']}/generate"

    def __del__(self):
        self._stop_server()

    def _gen_params_to_vllm_params(self, params: Dict) -> Dict:
        if "max_gen_len" in params:
            params["max_tokens"] = params["max_gen_len"]

        if "max_new_tokens" in params:
            params["max_tokens"] = params.pop("max_new_tokens")

        if "do_sample" in params and not params["do_sample"]:
            logger.info(f"do_sample is false, setting temperature to 0.")
            params["temperature"] = 0.0

        if "use_beam_search" in params and params["use_beam_search"]:
            logger.info(f"Beam search is enabled, setting temperature to 0.")
            params["temperature"] = 0.0

            if "best_of" not in params:
                logger.info(f"Beam search is enabled, setting best_of to 2.")
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
        params = self._gen_params_to_vllm_params(params)

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
        api_url = self._get_generate_uri()
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
            generated_text = self._del_prompt_if_req(prompt, output)
            inference_time_ms = (end_time - start_time) * 1000
            logger.info(f"Inference time: {inference_time_ms} ms")

            # TODO: Until mii returns the num tokens, approximate num_tokens. roughly, 75 words ~= 100 tokens
            num_tokens = (
                len(self._del_prompt_if_req(prompt, output, force=True).split(" "))
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


# Helper function to check if a port is open
def is_port_open(host: str = "0.0.0.0", port: int = 8000, timeout: float = 1.0) -> bool:
    """Check if a port is open on the given host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False
