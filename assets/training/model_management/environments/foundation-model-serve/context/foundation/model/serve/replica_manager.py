# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This class provides model replication and load balancing functionality."""
import os
import random
import tempfile
import time
import torch
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List
from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.constants import EnvironmentVariables, EngineName, CommonConstants
from foundation.model.serve.engine.engine import BaseEngine
from foundation.model.serve.engine.vllm_engine import VLLMEngine
from foundation.model.serve.engine.custom_engine import CustomEngine


logger = configure_logger(__name__)

def get_engine() -> BaseEngine:
    """Return the appropriate engine based on the engine name."""
    engine_name = os.getenv(EnvironmentVariables.ENGINE_NAME, EngineName.VLLM)
    startup_script_path = os.getenv(EnvironmentVariables.ENGINE_STARTUP_FILE_PATH, None)
    if startup_script_path:
        if os.path.isfile(startup_script_path):
            return CustomEngine()
        else:
            logger.error(f"Provided ENGINE_STARTUP_FILE_PATH {startup_script_path} does not exist. Falling back to {engine_name}.")

    if engine_name == EngineName.VLLM:
        return VLLMEngine()
    else:
        raise ValueError("Invalid engine name.")

@dataclass
class InferenceResult:
    """Data class for storing inference results."""
    response: str

class ReplicaManager:
    """Class for managing replicas of a model."""

    def __init__(self):
        """Initialize the ReplicaManager."""
        self.engine_replicas = []  # type: List[BaseEngine]
        self._replica_index = 0  # index of the next available replica
        self._tensor_parallel = int(os.getenv("AML_TENSOR_PARALLEL_SIZE", torch.cuda.device_count()))
        self.gpu_ids = self._get_cuda_visible_devices()

    @staticmethod
    def _get_cuda_visible_devices():
        """Get the CUDA_VISIBLE_DEVICES environment variable or set it to all available GPUs.

        Returns a comma-separated string of GPU IDs, e.g. "0,1,2,3"
        """
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if gpu_ids is None:
            gpu_ids = ",".join(map(str, range(torch.cuda.device_count()))) if torch.cuda.is_available() else ""
        return gpu_ids

    def initialize(self):
        """Initialize the ReplicaManager by creating engine replicas."""
        num_replicas=int(os.environ.get("NUM_REPLICAS", 1))
        for idx in list(range(num_replicas)):
            engine_replica = get_engine()
            self.engine_replicas.append(engine_replica)
        #self.engine_replicas[0].init_server()
        flag_file_path = os.path.join(tempfile.gettempdir(), "model_loaded_flag.txt")
        process_is_loading_model = False

        # wait a random amount of time between 1-5 seconds (in float), to avoid all workers trying to acquire
        # the lock at the same time
        time.sleep(random.uniform(0, 2))

        if os.path.exists(flag_file_path):
            logger.info(
                f"PID[{os.getpid()}] Model already being loaded by another worker.",
            )
            # wait for all replicas to finish loading the model
            while os.path.exists(flag_file_path):
                logger.info(f"Waiting for model to finish loading. Current worker pid: {os.getpid()}")
                time.sleep(5)
            for engine in self.engine_replicas:
                engine.wait_until_server_healthy(CommonConstants.HOST, "8000")
        else:
            try:
                with open(flag_file_path, "x"):
                    logger.info(
                        f"Lock acquired by worker with pid: {os.getpid()}. Loading model. \
Using tensor parallel of {self._tensor_parallel} GPUs per replica.",
                    )
                    process_is_loading_model = True
                    logger.handlers[0].flush()
                    os.environ["LOGGING_WORKER_ID"] = str(os.getpid())
                    with ThreadPoolExecutor() as executor:
                        self.engine_replicas = list(
                            executor.map(
                                self._launch_single_replica,
                                range(num_replicas),
                            )
                        )
            except FileExistsError:
                logger.info(
                    f"Model already being loaded by another worker. Waiting for model to finish loading. "
                    f"Current worker pid: {os.getpid()}",
                )

        if process_is_loading_model:
            # Load the model and print GPU information
            logger.info("###### GPU INFO ######")
            logger.info(os.system("nvidia-smi"))
            logger.info("###### GPU INFO ######")

            if os.path.exists(flag_file_path):
                os.remove(flag_file_path)

    def _launch_single_replica(self, replica_idx):
        """Launch a single replica."""
        engine_replica : BaseEngine = self.engine_replicas[replica_idx]
        gpu_ids_list = [int(gpu_id.strip()) for gpu_id in self.gpu_ids.split(",")]

        local_env = os.environ.copy()
        replica_gpu_ids = self._get_gpu_ids_for_replica(replica_idx, gpu_ids_list)
        cuda_visible_devices = ",".join(map(str, replica_gpu_ids))
        logger.debug(f"Setting CUDA_VISIBLE_DEVICES to {cuda_visible_devices} for replica {replica_idx + 1}")
        local_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        try:
            logger.info(
                f"Starting replica {replica_idx + 1} with GPUs {cuda_visible_devices}"
            )
            engine_replica.init_server()
            engine_replica.wait_until_server_healthy(CommonConstants.HOST, os.getenv(EnvironmentVariables.ENGINE_STARTUP_PORT, str(CommonConstants.DEFAULT_PORT)))
        except Exception as e:
            logger.error(f"Failed to start replica {replica_idx + 1} with GPUs {cuda_visible_devices}: {e}")
            traceback.print_exc()
        return engine_replica

    def _get_gpu_ids_for_replica(self, replica_idx: int, gpu_ids_list: List[int]) -> List[int]:
        """Get the GPU IDs for a specific replica."""
        # By default, use all available GPUs
        if self._tensor_parallel in ["", None]:
            replica_gpu_ids = gpu_ids_list
        else:
            # Determine the GPU IDs to use for this replica.
            start_gpu_idx = replica_idx * self._tensor_parallel
            end_gpu_idx = start_gpu_idx + self._tensor_parallel
            replica_gpu_ids = gpu_ids_list[start_gpu_idx:end_gpu_idx]
        return replica_gpu_ids

if __name__ == "__main__":
    replica_manager = ReplicaManager()
    replica_manager.initialize()

    print("Initialized replicas.")
