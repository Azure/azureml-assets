# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides classes and methods for handling inference tasks."""

import os
import socket
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from configs import EngineConfig

from dataclasses import dataclass

from constants import ServerSetupParams, TaskType
from logging_config import configure_logger
from utils import log_execution_time, box_logger

logger = configure_logger(__name__)


@dataclass
class InferenceResult:
    """Data class for storing inference results."""

    response: str
    inference_time_ms: float
    time_per_token_ms: float
    generated_tokens: List[Any]
    prompt_num: int
    error: Optional[str] = None

    def print_results(self):
        """Print the inference results of a single prompt."""
        if self.error:
            msg = f"## Inference Results ##\n ERROR: {self.error}"
        else:
            # TODO: record time per prompt in mii so we can show inference time for each prompt
            msg = f''' ## Prompt {self.prompt_num} Results ##\n Total Tokens Generated: {len(self.generated_tokens)}'''
        box_logger(msg)


class BaseEngine(ABC):
    """Base class for inference engines backends."""

    @abstractmethod
    def load_model(self, env=None):
        """Abstract method to load the model."""
        raise NotImplementedError("load_model method not implemented.")

    def init_client(self):
        """Initialize client[s] for the engine to receive requests on."""
        pass

    @property
    def server_url(self) -> str:
        """Return the server url for the engine."""
        return ""

    @abstractmethod
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Abstract method to generate responses for given prompts."""
        raise NotImplementedError("generate method not implemented.")

    def _del_prompt_if_req(
        self, prompt: str, response: str, params: Dict, force: bool = False
    ) -> str:
        """Delete the prompt from the response if required."""
        if force:
            return response[len(prompt):].strip()
        if self.task_config.task_type == TaskType.TEXT_GENERATION:
            if "return_full_text" in params and not params["return_full_text"]:
                return response[len(prompt):].strip()
            return response
        elif self.task_config.task_type == TaskType.CONVERSATIONAL:
            return response[len(prompt):].strip()
        else:
            raise ValueError(f"Invalid task type {self.task_config.task_type}.")

    # Helper function to check if a port is open
    def is_port_open(self, host: str = "localhost", port: int = 8000, timeout: float = 1.0) -> bool:
        """Check if a port is open on the given host."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    @log_execution_time
    def wait_until_server_healthy(self, host: str, port: int, timeout: float = 1.0):
        """Wait until the server is healthy."""
        start_time = time.time()
        while time.time() - start_time < ServerSetupParams.WAIT_TIME_MIN * 60:
            is_healthy = self.is_port_open(host, port, timeout)
            if is_healthy:
                if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                    logger.info("Server is healthy.")
                return
            if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
                logger.info("Waiting for server to start...")
            time.sleep(30)
        raise Exception("Server did not become healthy within 15 minutes.")

    @log_execution_time
    def get_tokens(self, response: str):
        """Load tokenizer and get tokens from a prompt."""
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.model_id)
        tokens = self.tokenizer.encode(response)
        return tokens


class HfEngine(BaseEngine):
    """Inference engine using Hugging Face methods."""

    def __init__(self, engine_config: EngineConfig):
        """Initialize the HfEngine with the given engine configuration."""
        self.engine_config = engine_config

    def load_model(self, env=None):
        """Load the model from the pretrained model specified in the engine configuration."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.engine_config.model_id)
        # move to the model to the GPU if testing on GPU
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.model.eval()

    @log_execution_time
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Generate responses for given prompts."""
        inference_results = []  # type: List[InferenceResult]
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                start_time = time.time()
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                output = self.model.generate(input_ids, max_new_tokens=20)
                generated_text = self.tokenizer.decode(
                    output[0], skip_special_tokens=True
                )
                inference_time_ms = (time.time() - start_time) * 1000
                num_tokens = len(output[0])
                time_per_token_ms = (
                    inference_time_ms / num_tokens if num_tokens > 0 else 0
                )
                result = InferenceResult(
                    response=generated_text,
                    inference_time_ms=inference_time_ms,
                    time_per_token_ms=time_per_token_ms,
                )
                inference_results.append(result)
        return inference_results
