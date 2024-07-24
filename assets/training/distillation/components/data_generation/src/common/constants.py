# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data generatior constants."""


# COMPONENT META
COMPONENT_NAME = "oss_distillation_generate_data"

# DATA GENERATOR VALIDATION
SUPPORTED_FILE_FORMATS = [".jsonl"]
TRAIN_FILE_NAME = "train_input.jsonl"
VALIDATION_FILE_NAME = "validation_input.jsonl"

SUPPORTED_TEACHER_MODEL_ASSET_IDS = [
    "azureml://registries/azureml-meta/models/Meta-Llama-3.1-405B-Instruct/versions/1"
]

# Scoring paths
VLLM_CHAT_SCORE_PATH = "/v1/chat/completions"
HFTV2_TEXT_GEN_SCORE_PATH = "/score"

# DATA GEN REQUEST
DEFAULT_SUCCESS_RATIO = 0.7
DEFAULT_REQUEST_BATCH_SIZE = 10
MAX_BATCH_SIZE = 100

# VLLM INFERENCE KEYS
TOP_P = "top_p"
MAX_TOKENS = "max_tokens"
MAX_NEW_TOKENS = "max_new_tokens"
TEMPERATURE = "temperature"
FREQUENCY_PENALTY = "frequency_penalty"
PRESENCE_PENALTY = "presence_penalty"
STOP_TOKEN = "stop"

# TEACHER MODEL DEFAULT INFERENCE PARAMS
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TOP_P = 0.1
DEFAULT_TEMPERATURE = 0.2

# CHAIN OF THOUGHT (COT)
COT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Write out in a step by step manner your reasoning about the answer using no more than 80 words. "
    "Based on the reasoning, produce the final answer. "
    "Your response should be in JSON format without using any backticks. "
    "The JSON is a dictionary whose keys are 'reason' and 'answer_choice'."
)


class InferenceMode:
    """Supported inference modes."""

    HFTV2_CHAT_COMPLETION = "hftv2_chat_completion"
    HFTV2_TEXT_GENERATION = "hftv2_text_generation"
    VLLM_CHAT_COMPLETION = "vllm_chat_completion"
    VLLM_TEXT_GENERATION = "vllm_text_generation"
