# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Mii Engine module.

This module contains the MiiEngineV2 class which is responsible for initializing the MII server and client,
generating responses for given prompts, and managing the allocation of processes and load balancing.
"""
import json
import math
import os
import time
from typing import Dict, List

import mii
import torch
from configs import EngineConfig, TaskConfig
from engine.engine import BaseEngine, InferenceResult
from logging_config import configure_logger
from mii.backend import MIIClient, MIIServer
from utils import log_execution_time

logger = configure_logger(__name__)


# TODO: Move them to mii config
MAX_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 4096))
DEVICE_COUNT = torch.cuda.device_count()


class MiiEngineV2(BaseEngine):
    """Inference engine using MII methods."""

    def __init__(self, config: EngineConfig, task_config: TaskConfig):
        """Initialize the MiiEngine with the given engine and task configurations."""
        self.engine_config = config
        self.task_config = task_config
        self.model = None
        self.mii_config = self._get_mii_config()

    def load_model(self, env=None):
        """Initialize MII server and MII client."""
        logger.info("MII Config: " + str(self.mii_config))
        logger.info("Start server setup")
        self.mii_server = MIIServer(self.mii_config)
        logger.info("Completed server setup")

    def init_client(self):
        """Initialize the MII client."""
        # wait until server is healthy then create client
        self.wait_until_server_healthy("localhost", self.mii_config.port_number)
        if self.model is None:
            self.model = MIIClient(self.mii_config)

    @log_execution_time
    async def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Generate responses for given prompts."""
        if self.model is None:
            logger.warning("MII client not initialized. Initializing now.")
            self.init_client()
        start_time = time.time()
        if isinstance(prompts, str):
            prompts = [prompts]
        try:
            responses = await self.model._request_async_response(prompts, **params)
        except Exception as e:
            raise Exception(
                json.dumps({"error": "Error in processing request", "exception": str(e)}))
        inference_time_ms = (time.time() - start_time) * 1000
        inference_results = []  # type: List[InferenceResult]
        for i, res in enumerate(responses):
            generated_text = res.generated_text
            response_tokens = self.get_tokens(generated_text)
            time_per_token_ms = inference_time_ms / len(response_tokens) if len(response_tokens) > 0 else 0

            result = InferenceResult(
                response=generated_text,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=time_per_token_ms,
                generated_tokens=response_tokens,
                prompt_num=i,
            )
            inference_results.append(result)
        return inference_results

    def _get_mii_config(self):
        """Get MII configuration."""
        model_size_in_gb = self._get_model_size_in_gb()

        if self.engine_config.tensor_parallel is not None:
            if self.engine_config.tensor_parallel > DEVICE_COUNT:
                raise ValueError(
                    f"TENSOR_PARALLEL ({self.engine_config.tensor_parallel}) is larger than the available GPUs",
                )
            replica_num = int(DEVICE_COUNT / self.engine_config.tensor_parallel)
        else:
            replica_num = self._calculate_replicas(model_size_in_gb)
            self.engine_config.tensor_parallel = int(DEVICE_COUNT / replica_num)

        default_mii_config = {
            "deployment_name": self.engine_config.mii_config.deployment_name,
            "deployment_type": mii.constants.DeploymentType.AML,
            "instance_type": "",  # this is only used for creating deployment script, can be left empty
            "model_config": {
                "all_rank_output": False,
                "inference_engine_config": {
                    "state_manager": {
                        "max_context": 8192,
                        "max_ragged_batch_size": 768,
                        "max_ragged_sequence_count": 512,
                        "max_tracked_sequences": 2048,
                        "memory_config": {"mode": "reserve", "size": 1000000000},
                        "offload": False,
                    },
                    "tensor_parallel": {"tp_size": self.engine_config.tensor_parallel},
                },
                "max_length": None,
                "model_name_or_path": self.engine_config.model_id,
                "profile_model_time": False,
                "replica_configs": [],
                "replica_num": replica_num,
                "sync_debug": False,
                "task": "text-generation",
                "tensor_parallel": self.engine_config.tensor_parallel,
                "tokenizer": self.engine_config.tokenizer,
            },
        }
        mii_config = mii.config.MIIConfig(**default_mii_config)
        return mii_config

    def _calculate_replicas(self, model_size_in_gb) -> int:
        """Calculate the number of replicas."""
        # Check GPU size and calculate number of replicas it can handle
        gpu_size_in_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        # For now, max 1 replica per 1 GPU
        # Taking in extra memory for cache
        # TODO: improve this logic based on the amount of KV cache required and token length
        num_possible_replicas = int(DEVICE_COUNT / math.ceil((model_size_in_gb / 0.8) / gpu_size_in_gb))
        if num_possible_replicas == 0:
            logger.debug(
                "Tensor parallel / model replica calculation with extra memory for cache "
                "results in 0 replicas. Calculating without extra memory for cache.",
            )
            num_possible_replicas = int(DEVICE_COUNT / math.ceil((model_size_in_gb) / gpu_size_in_gb))
            if num_possible_replicas == 0:
                raise ValueError("Not enough GPU to support model. Please select bigger SKU.")

        return num_possible_replicas

    def _get_model_size_in_gb(self) -> int:
        # Loop through the files in the folder
        file_ext = ""
        for file in os.listdir(self.engine_config.model_id):
            # Get the file extension
            curr_file_ext = os.path.splitext(file)[1]
            # Check if the file extension is .bin
            if curr_file_ext == ".bin" or curr_file_ext == ".safetensors":
                file_ext = curr_file_ext
                break
        # Initialize a variable to store the total size
        total_size = 0
        for file in os.listdir(self.engine_config.model_id):
            if file.endswith(file_ext):
                # Get the full path of the file
                file_path = os.path.join(self.engine_config.model_id, file)
                # Get the size of the file in bytes
                file_size = os.path.getsize(file_path)
                # Add the size to the total size
                total_size += file_size
        # Return the total size
        return math.ceil(total_size / (1024**3))

    async def shutdown_async(self):
        """Terminate DS-MII Server."""
        try:
            await self.model.terminate_async()
        except Exception as e:
            raise Exception(
                json.dumps({"error": "Error in processing request", "exception": str(e)}))
