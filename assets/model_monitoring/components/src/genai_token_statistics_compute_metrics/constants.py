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
COMPLETION_COUNT_KEY = "llm.token_count.completion"
PROMPT_COUNT_KEY = "llm.token_count.prompt"
TOTAL_COUNT_KEY = "     llm.token_count.total"
MODEL_KEY = "llm.model"
