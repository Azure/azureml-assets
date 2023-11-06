# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This module provides the FMScore class for running inference engines with given prompts and parameters."""

import os
from typing import Dict, List
from configs import EngineConfig, TaskConfig
from constants import TaskType
from engine.engine import InferenceResult
from engine.hf_engine import HfEngine
from managed_inference import MIRPayload
from prompt_formatter import Llama2Formatter
from utils import log_execution_time
from logging_config import configure_logger

logger = configure_logger(__name__)


def get_engine(engine_name: str, engine_config: EngineConfig, task_config: TaskConfig):
    """Return the appropriate engine based on the engine name."""
    if engine_name == "hf":
        return HfEngine(engine_config, task_config)
    elif engine_name == "vllm":
        from engine.vllm_engine import VLLMEngine

        return VLLMEngine(engine_config, task_config)
    elif engine_name == "mii":
        from engine.mii_engine import MiiEngine

        return MiiEngine(engine_config, task_config)
    else:
        raise ValueError("Invalid engine name.")


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
        self.engine = self._initialize_engine()
        self.formatter = self._initialize_formatter()

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
        print(f"Formatted prompts: {formatted_prompts}")
        inference_results = self.engine.generate(formatted_prompts, payload.params)
        logger.info(
            f"inference_time_ms: {inference_results[0].inference_time_ms}, "
            f"time_per_token_ms: {inference_results[0].time_per_token_ms}"
        )
        return inference_results

    def _initialize_engine(self):
        print(f"Initializing engine: {self.engine_config.engine_name}")
        self.engine = get_engine(
            self.engine_config.engine_name, self.engine_config, self.task_config
        )

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
        flag_file_path = "/tmp/model_loaded_flag.txt"
        process_is_loading_model = False
        if os.path.exists(flag_file_path):
            logger.info(
                f"Model already loaded by another worker. Current worker pid: {os.getpid()}"
            )
        else:
            try:
                with open(flag_file_path, "x"):
                    logger.info(
                        f"Lock acquired by worker with pid: {os.getpid()}. Loading model."
                    )
                    process_is_loading_model = True
                    self.engine.load_model()
                    logger.info(f"Model loaded by worker with pid: {os.getpid()}")
            except FileExistsError:
                logger.info(
                    f"Model already being loaded by another worker. Waiting for model to finish loading. "
                    f"Current worker pid: {os.getpid()}"
                )

        self.engine.init_client()  # wait for the model to finish loading
        if process_is_loading_model:
            # run nvidia-smi to check GPU usage after the model is loaded
            logger.info("###### GPU INFO ######")
            logger.info(os.system("nvidia-smi"))
            logger.info("###### GPU INFO ######")
            if os.path.exists(flag_file_path):
                os.remove(flag_file_path)

        return self.engine

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
