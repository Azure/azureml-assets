# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains constants for genai token statistics components."""

INCLUDE_SPAN_TYPE = ["LLM", "Embedding"]

# Constants for metric count
TOTAL_TOKEN_COUNT = "TotalTokenCount"
TOTAL_PROMPT_COUNT = "TotalPromptCount"
TOTAL_COMPLETION_COUNT = "TotalCompletionCount"
AVG_TOKEN_COUNT = "AvgTokenCount"
AVG_PROMPT_COUNT = "AvgPromptCount"
AVG_COMPLETION_COUNT = "AvgCompletionCount"
MODEL_COMPLETION_COUNT = "model_completion_count"

# Constants for attribute keys
COMPLETION_COUNT_KEYS = ["llm.token_count.completion", "llm.usage.completion_tokens"]
PROMPT_COUNT_KEYS = ["embedding.usage.prompt_tokens", "llm.token_count.prompt",
                     "llm.usage.prompt_tokens", "embedding.token_count.prompt"]
TOTAL_COUNT_KEYS = ["embedding.usage.total_tokens", "llm.token_count.total", "embedding.token_count.total",
                    "llm.usage.total_tokens"]
MODEL_KEYS = ["llm.response.model", "embedding.response.model", "llm.model", "embedding.model"]
