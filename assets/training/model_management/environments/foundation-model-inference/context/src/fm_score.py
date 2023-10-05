from typing import Dict, List
from configs import EngineConfig, TaskConfig
from constants import TaskType
from engine.engine import HfEngine, InferenceResult
from prompt_formatter import Llama2Formatter
from utils import log_execution_time
from logging_config import configure_logger

logger = configure_logger(__name__)


def get_engine(engine_name: str, engine_config: EngineConfig, task_config: TaskConfig):
    if engine_name == "hf":
        return HfEngine(engine_config)
    elif engine_name == "vllm":
        from engine.vllm_engine import VLLMEngine

        return VLLMEngine(engine_config, task_config)
    elif engine_name == "mii":
        from engine.mii_engine import MiiEngine

        return MiiEngine(engine_config, task_config)
    else:
        raise ValueError("Invalid engine name.")


def get_formatter(model_name: str):
    if model_name == "Llama2":
        return Llama2Formatter()
    else:
        raise ValueError("Invalid model name.")


class FMScore:
    def __init__(self, config: Dict):
        self.task_config = TaskConfig.from_dict(config["task"])
        self.engine_config = EngineConfig.from_dict(config["engine"])

    def init(self):
        self.engine = self._initialize_engine()
        self.formatter = self._initialize_formatter()

    @log_execution_time
    def run(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """
        Run the engine with the given prompts and parameters.
        :param prompts: List of prompts
        :param params: Dictionary of parameters
        :return: A list of InferenceResult objects, each containing the response and metadata related to the inference
        """
        formatted_prompts = [
            self.formatter.format_prompt(self.task_config.task_type, prompt, params)
            for prompt in prompts
        ]
        print(f"Formatted prompts: {formatted_prompts}")
        inference_results = self.engine.generate(formatted_prompts, params)
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
        self.engine.load_model()
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
    fms.run("Today is a wonderful day to ", {"max_length": 128})
