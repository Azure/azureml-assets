from engine.engine import AbstractEngine, InferenceResult
from utils import log_execution_time
from configs import EngineConfig, TaskConfig
from logging_config import configure_logger
from engine._hf_predictors import get_predictor
from constants import TaskType
import pandas as pd

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, pipeline

import torch
import time

logger = configure_logger(__name__)

def sanitize_load_args(items):
    for item in items:
        if isinstance(items[item], str) and items[item].startswith("torch."):
            items[item] = eval(items[item])
    return items


class HfEngine(AbstractEngine):
    """Inference engine using Hugging Face methods."""

    def __init__(self, engine_config: EngineConfig, task_config: TaskConfig):
        """Initialize the HfEngine with the given engine configuration."""
        self.engine_config = engine_config
        self.task_config = task_config

    def load_model(self):
        """Load the model from the pretrained model specified in the engine configuration."""
        if self.engine_config.model_kwargs.get("device_map", None) or self.engine_config.model_kwargs.get("device", None):
            self.engine_config.model_kwargs["device_map"] = "auto"
        self.config = AutoConfig.from_pretrained(self.engine_config.hf_config_path,
                                                 **sanitize_load_args(self.engine_config.config_kwargs))
        self.tokenizer = AutoTokenizer.from_pretrained(self.engine_config.hf_tokenizer_path,
                                                       **sanitize_load_args(self.engine_config.tokenizer_kwargs))
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.engine_config.model_id, config=self.config,
                                                              **sanitize_load_args(self.engine_config.model_kwargs))
        except ValueError:
            logger.info("Model doesn't support device_map. Trying with device instead.")
            self.engine_config.model_kwargs["device"] = torch.cuda.current_device() if torch.cuda.is_available() else -1
            self.model = AutoModelForCausalLM.from_pretrained(self.engine_config.model_id, config=self.config,
                                                              **sanitize_load_args(self.engine_config.model_kwargs))
        except Exception as e:
            logger.warning(f"Failed to load the model with exception: {repr(e)}")
            raise e
        
    def _get_problem_type(self):
        if hasattr(self.config, "problem_type"):
            return self.config.problem_type
        return None

    @log_execution_time
    def generate(self, prompts: List[str], params: Dict) -> List[InferenceResult]:
        """Generate responses for given prompts."""
        inference_results = []  # type: List[InferenceResult]
        df = pd.DataFrame(prompts)
        st = time.time()
        predictor_cls = get_predictor(task_type=self.task_config.task_type, problem_type=self._get_problem_type())
        predictor_obj = predictor_cls(task_type=self.task_config.task_type, model=self.model,
                                      tokenizer=self.tokenizer, config=self.config)
        if self.task_config.task_type in (TaskType.TEXT_GENERATION, TaskType.CONVERSATIONAL, TaskType.CHAT_COMPLETION):
            preds = predictor_obj.predict(df, generator_config=params)
        else:
            preds = predictor_obj.predict(df, tokenizer_config=params)
        et = time.time()
        inference_time_ms = (et - st) * 1000 / len(prompts)
        num_tokens = len(preds[0])
        time_per_token_ms = (
            inference_time_ms / num_tokens if num_tokens > 0 else 0
        )
        result = [InferenceResult(
            response=generated_text,
            inference_time_ms=inference_time_ms,
            time_per_token_ms=time_per_token_ms,
        ) for generated_text in preds]
        inference_results.append(result)
        return inference_results