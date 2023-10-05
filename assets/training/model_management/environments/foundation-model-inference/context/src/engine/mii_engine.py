# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from configs import EngineConfig, TaskConfig
from engine.engine import AbstractEngine, InferenceResult
import os
import torch
import time
import mii
from mii.config import LoadBalancerConfig, ReplicaConfig
from typing import Dict, List
from logging_config import configure_logger
from utils import log_execution_time

logger = configure_logger(__name__)


# TODO: Move them to mii config
LOAD_BALANCING_PORT = 50050
MAX_TOKENS = int(os.environ.get("MAX_TOTAL_TOKENS", 4096))
TORCH_DIST_PORT = 29501
REPLICA_NUM = int(os.getenv("WORKER_COUNT", 1))
DEVICE_COUNT = torch.cuda.device_count()
MODEL_DIR = os.getenv("AZUREML_MODEL_DIR", "")
MODEL_PATH = "mlflow_model_folder/data/model"


class MiiEngine(AbstractEngine):
    def __init__(self, config: EngineConfig, task_config: TaskConfig):
        self.engine_config = config
        self.task_config = task_config
        self.mii_config = self.engine_config.mii_config
        self.model = None

    def load_model(self):
        """Initialize MII server and MII client."""
        load_balancer_config = self._generate_load_balancer_config()
        is_70b_model = "Llama-2-70b" in MODEL_DIR or "Llama-2-70b-chat" in MODEL_DIR
        replace_with_kernel_inject = not is_70b_model
        default_mii_configs = {
            "checkpoint_dict": None,
            "deploy_rank": load_balancer_config.replica_configs[0].gpu_indices,
            "dtype": torch.float16,
            "enable_cuda_graph": False,
            "enable_restful_api": False,
            "hf_auth_token": None,
            "load_with_sys_mem": True,
            "max_tokens": MAX_TOKENS,
            "meta_tensor": False,
            "port_number": LOAD_BALANCING_PORT,
            "profile_model_time": False,
            "replace_with_kernel_inject": replace_with_kernel_inject,
            "replica_num": REPLICA_NUM,
            "skip_model_check": True,
            "tensor_parallel": self.engine_config.tensor_parallel,
            "torch_dist_port": TORCH_DIST_PORT,
            "trust_remote_code": False,
        }
        configs = {
            "deployment_name": self.mii_config.deployment_name,
            "ds_config": self.mii_config.ds_config,
            "ds_optimize": self.mii_config.ds_optimize,
            "ds_zero": self.mii_config.ds_zero,
            "load_balancer_config": load_balancer_config,
            "mii_configs": {**default_mii_configs, **self.mii_config.mii_configs},
            # TODO: Change this to model_id
            "model_name": MODEL_DIR,
            "model_path": MODEL_PATH,
            # TODO: Use self.task_config.task_type, figure out why 'conversational' isn't working
            "task_name": "text-generation",
        }

        deployment_name = configs["deployment_name"]
        model_path = mii.utils.full_model_path(MODEL_PATH)
        model_name = configs["model_name"]
        task_name = configs["task_name"]

        start_server = True
        if int(os.getpid()) % configs.get("mii_configs").get("replica_num") != 0:
            start_server = False
            logger.info("Skip MII server setup for this process")

        if start_server:
            logger.info("Start server setup")
            self.mii_server = mii.MIIServer(
                deployment_name,
                task_name,
                model_name,
                model_path,
                ds_optimize=configs["ds_optimize"],
                ds_zero=configs["ds_zero"],
                ds_config=configs["ds_config"],
                mii_configs=configs["mii_configs"],
                lb_config=load_balancer_config,
            )
            logger.info("Completed server setup")
            time.sleep(20)

        self.model = mii.MIIClient(
            task_name, "localhost", configs["mii_configs"]["port_number"]
        )

    @log_execution_time
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Call the model to get the text generation or chat completion results."""
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

    def _allocate_processes(self, hostfile_path):
        from mii.server import _allocate_processes

        if hostfile_path is None:
            import tempfile

            hostfile_path = tempfile.NamedTemporaryFile(delete=False).name
            print(f"hostfile_path: {hostfile_path}")
            num_gpu = DEVICE_COUNT
            with open(hostfile_path, "w") as f:
                f.write(f"localhost slots={num_gpu}")
        return _allocate_processes(
            hostfile_path, self.engine_config.tensor_parallel, REPLICA_NUM
        )

    def _generate_load_balancer_config(self):
        replica_pool = self._allocate_processes(hostfile_path=None)
        replica_configs = [
            ReplicaConfig(
                hostname=hostname,
                tensor_parallel_ports=list(
                    range(
                        LOAD_BALANCING_PORT
                        + i * self.engine_config.tensor_parallel
                        + 1,
                        LOAD_BALANCING_PORT
                        + i * self.engine_config.tensor_parallel
                        + 1
                        + self.engine_config.tensor_parallel,
                    )
                ),
                torch_dist_port=i + TORCH_DIST_PORT,
                gpu_indices=gpu_indices,
            )
            for i, (hostname, gpu_indices) in enumerate(replica_pool)
        ]
        load_balancer_config = LoadBalancerConfig(
            port=LOAD_BALANCING_PORT, replica_configs=replica_configs
        )
        return load_balancer_config
