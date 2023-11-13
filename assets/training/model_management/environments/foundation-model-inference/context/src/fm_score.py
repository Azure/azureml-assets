# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides the FMScore class for running inference engines with given prompts and parameters."""

import os
from typing import Dict, List
from configs import EngineConfig, TaskConfig
from constants import TaskType
from engine.engine import InferenceResult
from managed_inference import MIRPayload
from prompt_formatter import Llama2Formatter
from replica_manager import ReplicaManagerConfig, ReplicaManager
from utils import log_execution_time
from logging_config import configure_logger

logger = configure_logger(__name__)


def get_formatter(model_name: str):
    """Return the appropriate formatter based on the model name."""
    if model_name == "Llama2":
        return Llama2Formatter()
    else:
        raise ValueError("Invalid model name.")


class FMScore:
    """Class for running inference engines with given prompts and parameters."""

    def __init__(self, config: Dict):
        """Initialize the FMScore with the given configuration."""
        self.task_config = TaskConfig.from_dict(config["task"])
        self.engine_config = EngineConfig.from_dict(config["engine"])

    def init(self):
        """Initialize the engine and formatter."""
        self.replica_manager = self._init_replica_manager()
        self.formatter = self._initialize_formatter()

    def _init_replica_manager(self):
        replica_manager_config = ReplicaManagerConfig(
            engine_config=self.engine_config,
            task_config=self.task_config,
            num_replicas=int(os.environ.get("NUM_REPLICAS", -1))
        )
        replica_manager = ReplicaManager(replica_manager_config)
        replica_manager.initialize()
        return replica_manager

    @log_execution_time
    def run(self, payload: MIRPayload) -> List[InferenceResult]:
        """
        Run the engine with the given prompts and parameters.

        :param payload: The parsed input from managed inference that contains the parameters and prompts from the user
        :return: A list of InferenceResult objects, each containing the response and metadata related to the inference
        """
        formatted_prompts = [
            self.formatter.format_prompt(self.task_config.task_type, prompt, payload.params)
            for prompt in payload.query
        ]
        inference_results = self.replica_manager.get_replica().generate(formatted_prompts, payload.params)
        return inference_results

    def _initialize_formatter(self):
        formatter = get_formatter(model_name="Llama2")
        return formatter


if __name__ == "__main__":
    # sample_config = {
    #     "engine": {
    #         # "engine_name": "HuggingFace",
    #         "engine_name": "vllm",
    #         "model_id": "gpt2",
    #     },
    #     "task": {
    #         "task_type": TaskType.TEXT_GENERATION,
    #     },
    # }

    sample_config = {
        "engine": {
            "engine_name": "mii",
            "model_id": "gpt2",
            "mii_config": {
                "deployment_name": "sample_deployment",
                "mii_configs": {
                    # configurations here
                },
                "model_path": "gpt2",
                "task_name": "text-generation",
                "ds_config": None,
                "ds_optimize": True,
                "ds_zero": False,
            },
        },
        "task": {
            "task_type": TaskType.TEXT_GENERATION,
        },
    }

    fms = FMScore(sample_config)
    fms.init()
    fms.run(MIRPayload("Today is a wonderful day to ", {"max_length": 128}, fms.task_config.task_type))
