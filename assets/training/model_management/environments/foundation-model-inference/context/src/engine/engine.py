# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides classes and methods for handling inference tasks."""

import socket
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from configs import EngineConfig

from dataclasses import dataclass

from constants import ServerSetupParams, TaskType
from logging_config import configure_logger
from utils import log_execution_time

logger = configure_logger(__name__)


@dataclass
class InferenceResult:
    """Data class for storing inference results."""

    response: str
    inference_time_ms: float
    time_per_token_ms: float
    error: Optional[str] = None


class AbstractEngine(ABC):
    """Abstract base class for inference engines."""

    @abstractmethod
    def load_model(self):
        """Abstract method to load the model."""
        raise NotImplementedError("load_model method not implemented.")

    def init_client(self):
        """Initialize client[s] for the engine to receive requests on."""
        pass

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
    def _is_port_open(self, host: str = "localhost", port: int = 8000, timeout: float = 1.0) -> bool:
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
            is_healthy = self._is_port_open(host, port, timeout)
            if is_healthy:
                logger.info("Server is healthy.")
                return
            logger.info("Waiting for server to start...")
            time.sleep(30)
        raise Exception("Server did not become healthy within 15 minutes.")


class HfEngine(AbstractEngine):
    """Inference engine using Hugging Face methods."""

    def __init__(self, engine_config: EngineConfig):
        """Initialize the HfEngine with the given engine configuration."""
        self.engine_config = engine_config

    def load_model(self):
        """Load the model from the pretrained model specified in the engine configuration."""
        self.config = AutoConfig.from_pretrained(self.engine_config.hf_config_path, trust_remote_code=self.engine_config.trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.hf_tokenizer_path, **self.engine_config.tokenizer_kwargs)
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
