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
        # TODO: Remove this once DS-Inference supports 70b models
        is_70b_model = "Llama-2-70b" in MODEL_DIR or "Llama-2-70b-chat" in MODEL_DIR
        replace_with_kernel_inject = not is_70b_model

        # TODO: Update to check whether meta tensor is supported for the model
        meta_tensor = False
        model_size_in_gb = self._get_model_size_in_gb()
        self._check_memory_requirement(model_size_in_gb, meta_tensor)

        if self.engine_config.tensor_parallel is not None:
            if self.engine_config.tensor_parallel > DEVICE_COUNT:
                raise ValueError(
                    f"TENSOR_PARALLEL ({self.engine_config.tensor_parallel}) is larger than the available GPUs"
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
                "checkpoint_dict": None,
                "deploy_rank": list(range(self.engine_config.tensor_parallel)),
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
                "tensor_parallel": self.engine_config.tensor_parallel,
                "trust_remote_code": False
            }
        }
        mii_config = mii.MIIConfig(**default_mii_config)
        return mii_config

    def _check_memory_requirement(self, model_size_in_gb, meta_tensor):
        """Check if the model can be loaded in the available memory.

        If loading with system memory, DeepSpeed-Inference requires RAM size of model_size * DEVICE_COUNT.
        If meta_tensor is enabled instead for the model, then skip this check.
        """
        # Check RAM required for loading the model
        # If meta tensor is available and if so, skip RAM check.
        if not meta_tensor:
            total_ram_in_gb = psutil.virtual_memory().total / (1024 ** 3)
            total_required_ram_in_gb = model_size_in_gb * DEVICE_COUNT
            if total_ram_in_gb < total_required_ram_in_gb:
                raise ValueError("Total RAM is smaller than required RAM.")

    def _calculate_replicas(self, model_size_in_gb) -> int:
        """Calculate the number of replicas."""
        # Check GPU size and calculate number of replicas it can handle
        gpu_size_in_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        # For now, max 1 replica per 1 GPU
        # Taking in extra memory for cache
        # TODO: improve this logic based on the amount of KV cache required and token length
        num_possible_replicas = int(DEVICE_COUNT/math.ceil((model_size_in_gb/0.8)/gpu_size_in_gb))
        if num_possible_replicas == 0:
            logger.debug(
                "Tensor parallel / model replica calculation with extra memory for cache "
                "results in 0 replicas. Calculating without extra memory for cache."
            )
            num_possible_replicas = int(DEVICE_COUNT/math.ceil((model_size_in_gb)/gpu_size_in_gb))
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
        return math.ceil(total_size / (1024 ** 3))
