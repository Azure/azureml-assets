# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This class provides model replication and load balancing functionality."""
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import List

import torch
from configs import EngineConfig, TaskConfig
from engine import BaseEngine
from logging_config import configure_logger

logger = configure_logger(__name__)


def get_engine(engine_config: EngineConfig, task_config: TaskConfig) -> BaseEngine:
    """Return the appropriate engine based on the engine name."""
    engine_name = engine_config.engine_name
    if engine_name == "hf":
        from engine import HfEngine

        return HfEngine(engine_config)
    elif engine_name == "vllm":
        from engine.vllm_engine import VLLMEngine

        return VLLMEngine(engine_config, task_config)
    elif engine_name == "mii":
        from engine.mii_engine_v2 import MiiEngineV2

        return MiiEngineV2(engine_config, task_config)
    elif engine_name == "mii-v1":
        from engine.mii_engine import MiiEngine

        return MiiEngine(engine_config, task_config)
    else:
        raise ValueError("Invalid engine name.")


@dataclass
class ReplicaManagerConfig:
    """Data class for storing the configuration of a ReplicaManager."""

    engine_config: EngineConfig  # homogeneous config for all replicas
    task_config: TaskConfig
    num_replicas: int
    gpu_ids: str = ""

    def __post_init__(self):
        """Initialize the ReplicaManagerConfig."""
        if self.gpu_ids == "":
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


class ReplicaManager:
    """Class for managing replicas of a model."""

    def __init__(self, replica_config: ReplicaManagerConfig):
        """Initialize the ReplicaManager."""
        self._replica_config = replica_config
        self._engine_config = self._replica_config.engine_config
        self._task_config = self._replica_config.task_config

        self.engine_replicas = []  # type: List[BaseEngine]
        self._replica_index = 0  # index of the next available replica

        self._set_defaults()

    def initialize(self):
        """Initialize the ReplicaManager by creating engine replicas."""
        num_replicas = self._replica_config.num_replicas
        # create engine replicas
        for idx in list(range(num_replicas)):
            engine_config = deepcopy(self._replica_config.engine_config)
            engine_config.tensor_parallel = self._tensor_parallel
            engine_config.port = self._replica_config.engine_config.port + idx
            engine_replica = get_engine(engine_config, self._task_config)
            self.engine_replicas.append(engine_replica)

        """
        This block of code is used to handle the loading of the model in a multi-process environment.
        A flag file at "/tmp/model_loaded_flag.txt" is used as a lock to ensure that the model is loaded only once.
        If the flag file exists, it means the model is already being loaded by another worker.
        If the flag file does not exist, the current worker creates the flag file and loads the model.
        If the flag file creation fails due to FileExistsError, it means another worker is currently loading
        the model.
        The current worker then waits for the model to finish loading.
        After the model has been loaded and the client has been initialized, the flag file is removed,
        acting as releasing the lock.
        """
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
                time.sleep(5)
            for engine in self.engine_replicas:
                engine.init_client()
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
                    # Start replicas in parallel using ProcessPoolExecutor
                    with ProcessPoolExecutor() as executor:
                        executor.map(
                            self._launch_single_replica,
                            range(num_replicas),
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
        if os.environ.get("LOGGING_WORKER_ID", "") == str(os.getpid()):
            print(f"Initialized {self._replica_config.num_replicas} replicas.")
            print(f"Server URIs: {[engine.server_url for engine in self.engine_replicas]}")

    def _launch_single_replica(self, replica_idx):
        """Launch a single replica."""
        engine_replica = self.engine_replicas[replica_idx]
        gpu_ids_list = [int(gpu_id.strip()) for gpu_id in self._replica_config.gpu_ids.split(",")]

        local_env = os.environ.copy()
        replica_gpu_ids = self._get_gpu_ids_for_replica(replica_idx, gpu_ids_list)
        cuda_visible_devices = ",".join(map(str, replica_gpu_ids))
        logger.debug(f"Setting CUDA_VISIBLE_DEVICES to {cuda_visible_devices} for replica {replica_idx + 1}")
        local_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        try:
            logger.info(
                f"Starting replica {replica_idx + 1} with GPUs {cuda_visible_devices} "
                f"for engine: {engine_replica.engine_config.engine_name}",
            )
            engine_replica.load_model(env=local_env)
            engine_replica.init_client()
        except Exception as e:
            logger.error(f"Failed to start replica {replica_idx + 1} with GPUs {cuda_visible_devices}: {e}")

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

    def get_replica(self) -> BaseEngine:
        """Return the next available replica based on round-robin."""
        replica = self.engine_replicas[self._replica_index]
        self._replica_index = (self._replica_index + 1) % len(self.engine_replicas)  # Next replica index
        logger.info(
            f"Returning replica {self._replica_index} with server URI {replica.server_url} as the next "
            f"available replica.",
        )
        return replica

    def _get_tensor_parallel(self):
        """Get the tensor parallel configuration for each replica."""
        # evenly divide the available GPUs among the replicas
        res = max(torch.cuda.device_count() // self._replica_config.num_replicas, 1)
        return res

    def _set_defaults(self):
        """Do some sanity checks and set default values for the replica config and tensor parallel."""
        if self._replica_config.num_replicas >= 1:
            # num_replicas is set by the user
            if self._replica_config.engine_config.tensor_parallel in [None, ""]:
                raise ValueError(
                    "Tensor parallel must be specified when using multiple replicas. "
                    "Set it using environment variable 'TENSOR_PARALLEL'.",
                )

            total_gpus_needed = self._replica_config.num_replicas * self._replica_config.engine_config.tensor_parallel
            gpu_ids_list = [int(gpu_id.strip()) for gpu_id in self._replica_config.gpu_ids.split(",")]
            if total_gpus_needed > len(gpu_ids_list):
                raise ValueError(
                    f"Insufficient GPUs: Need {total_gpus_needed} but only {len(gpu_ids_list)} are available. "
                    f"Reduce NUM_REPLICAS or TENSOR_PARALLEL to fit within the available GPUs.",
                )
        else:
            # use 1 replica by default, if not specified by the user
            self._replica_config.num_replicas = 1

        self._tensor_parallel = (
            self._replica_config.engine_config.tensor_parallel
            if self._replica_config.engine_config.tensor_parallel
            else self._get_tensor_parallel()
        )


if __name__ == "__main__":
    engine_config = EngineConfig(
        engine_name="vllm",
        model_id="/data/Llama-2-7b-chat/mlflow_model_folder/data/model/",
        tensor_parallel=1,
    )

    replica_config = ReplicaManagerConfig(
        engine_config=engine_config,
        task_config=TaskConfig(),
        num_replicas=4,
    )

    replica_manager = ReplicaManager(replica_config)
    replica_manager.initialize()

    print("Initialized replicas.")

    prompt = "The meaning of life is"
    max_new_tokens = 500

    for i in range(replica_config.num_replicas):
        engine_replica = replica_manager.get_replica()
        print(f"Replica {i + 1} server URI: {engine_replica.server_url}")
        print(engine_replica.generate([prompt], {"max_new_tokens": max_new_tokens}))
        print()

    while True:
        time.sleep(1)
