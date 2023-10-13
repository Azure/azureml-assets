# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mii Engine module.

This module contains the MiiEngine class which is responsible for initializing the MII server and client,
generating responses for given prompts, and managing the allocation of processes and load balancing.
"""

from configs import EngineConfig, TaskConfig
from engine.engine import AbstractEngine, InferenceResult
import math
import os
import psutil
import torch
import time
import mii
from typing import Dict, List
from logging_config import configure_logger
from utils import log_execution_time

logger = configure_logger(__name__)


# TODO: Move them to mii config
MAX_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 4096))
REPLICA_NUM = os.environ.get("REPLICA_NUM", None)
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR", "")
MODEL_PATH = "mlflow_model_folder/data/model"
DEVICE_COUNT = torch.cuda.device_count()


class MiiEngine(AbstractEngine):
    """Inference engine using MII methods."""

    def __init__(self, config: EngineConfig, task_config: TaskConfig):
        """Initialize the MiiEngine with the given engine and task configurations."""
        self.engine_config = config
        self.task_config = task_config
        self.model = None
        self.mii_config = self._get_mii_config()

    def load_model(self):
        """Initialize MII server and MII client."""
        logger.info("MII Config: " + str(self.mii_config))
        logger.info("Start server setup")
        self.mii_server = mii.MIIServer(self.mii_config)
        logger.info("Completed server setup")

    def init_client(self):
        """Initialize the MII client."""
        # wait until server is healthy then create client
        self.wait_until_server_healthy("localhost", self.mii_config.port_number)
        if self.model is None:
            self.model = mii.MIIClient(
                self.mii_config.model_config.task, "localhost", self.mii_config.port_number
            )

    @log_execution_time
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Generate responses for given prompts."""
        assert (
            self.model is not None
        ), "MII client not initialized. Please call init_client() before calling generate()"
        queries = {"query": prompts}
        start_time = time.time()
        responses = self.model.query(queries, **params)
        inference_time_ms = (time.time() - start_time) * 1000
        inference_results = []  # type: List[InferenceResult]
        for i, res in enumerate(responses.response):
            generated_text = res
            generated_text = self._del_prompt_if_req(prompts[i], generated_text)
            # TODO: Until mii returns the num tokens, approximate num_tokens. roughly, 75 words ~= 100 tokens
            num_tokens = (
                len(
                    self._del_prompt_if_req(
                        prompts[i], generated_text, force=True
                    ).split(" ")
                )
                // 75
                * 100
            )
            time_per_token_ms = inference_time_ms / num_tokens if num_tokens > 0 else 0
            result = InferenceResult(
                response=generated_text,
                inference_time_ms=inference_time_ms,
                time_per_token_ms=time_per_token_ms,
            )
            inference_results.append(result)
        return inference_results

    def _get_mii_config(self):
        """Get MII configuration."""
        is_70b_model = "Llama-2-70b" in MODEL_DIR or "Llama-2-70b-chat" in MODEL_DIR
        replace_with_kernel_inject = not is_70b_model
        replica_num = self._calculate_replicas()
        tensor_parallel = int(DEVICE_COUNT / replica_num)

        default_mii_config = {
            "deployment_name": self.engine_config.mii_config.deployment_name,
            "deployment_type": mii.constants.DeploymentType.AML,
            "instance_type": "",  # this is only used for creating deployment script, can be left empty
            "model_config": {
                "checkpoint_dict": None,
                "deploy_rank": list(range(tensor_parallel)),
                "ds_config": self.engine_config.mii_config.ds_config,
                "dtype": torch.float16,
                "enable_cuda_graph": False,
                "enable_deepspeed": self.engine_config.mii_config.enable_deepspeed,
                "enable_zero": self.engine_config.mii_config.ds_zero,
                "hf_auth_token": None,
                "load_with_sys_mem": True,
                "max_tokens": MAX_TOKENS,
                "meta_tensor": False,
                "model": MODEL_DIR,
                "model_path": MODEL_PATH,
                "profile_model_time": False,
                "replace_with_kernel_inject": replace_with_kernel_inject,
                "replica_configs": [],
                "replica_num": replica_num,
                "skip_model_check": True,
                # "task": self.task_config.task_type,
                "task": "text-generation",
                "tensor_parallel": tensor_parallel,
                "trust_remote_code": False
            }
        }
        mii_config = mii.MIIConfig(**default_mii_config)
        return mii_config

    def _calculate_replicas(self) -> int:
        """Calculate the number of replicas."""
        # if REPLICA_NUM is set, use that
        if REPLICA_NUM is not None:
            if int(REPLICA_NUM) <= DEVICE_COUNT:
                return int(REPLICA_NUM)
            else:
                logger.warning(
                    f"REPLICA_NUM ({REPLICA_NUM}) is larger than the number of GPUs ({DEVICE_COUNT}). "
                    f"Proceeding to calculate the number of replicas based on the model size."
                )

        # Check RAM required for loading the model
        # TODO: Check if meta tensor is available and if so, skip RAM check.
        device_count = torch.cuda.device_count()
        total_ram_in_gb = psutil.virtual_memory().total / (1024 ** 3)
        model_size_in_gb = self._get_model_size()
        total_required_ram_in_gb = model_size_in_gb * device_count
        if total_ram_in_gb < total_required_ram_in_gb:
            raise ValueError("Total RAM is smaller than required RAM.")
        # Check GPU size and calculate number of replicas it can handle
        gpu_size_in_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # For now, max 1 replica per 1 GPU
        # Taking in extra memory for cache
        # TODO: improve this logic based on the amount of KV cache required and token length
        num_possible_replicas = int(device_count/math.ceil((model_size_in_gb/0.8)/gpu_size_in_gb))
        if num_possible_replicas == 0:
            raise ValueError("Not enough GPU to support model. Please select bigger SKU.")
        return num_possible_replicas

    def _get_model_size(self) -> int:
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
        return math.ceil(total_size / (1024 ** 3))
