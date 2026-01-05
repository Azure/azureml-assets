# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Configuration and utilities for mocking evaluator behavior in tests.

Defines output formats, scores, and side effects for different evaluator types.
"""

from enum import Enum
from typing import Callable, Awaitable, Dict, Any


# Constants for success scores
GRADERS_SUCCESS_SCORE = 5
BINARY_SUCCESS_SCORE = 1
DEFAULT_EXPLANATION = "Success explanation"


class OutputType(Enum):
    """Types of outputs that evaluators can produce."""

    STRING = "string"
    DICT = "dict"


class EvaluatorCategory(Enum):
    """Categories of evaluators based on their scoring system."""

    GRADERS = "graders"  # Uses 1-5 scale
    BINARY = "binary"  # Uses 0-1 scale


class EvaluatorOutputConfig:
    """Configuration for an evaluator's output format and scoring."""

    def __init__(self, category: EvaluatorCategory, output_type: OutputType):
        """Initialize configuration with category and output type."""
        self.category = category
        self.output_type = output_type
        self.score = GRADERS_SUCCESS_SCORE if category == EvaluatorCategory.GRADERS else BINARY_SUCCESS_SCORE


# Mapping of evaluator names to their output configurations
EVALUATOR_CONFIGS: Dict[str, EvaluatorOutputConfig] = {
    "fluency": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.STRING),
    "coherence": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.STRING),
    "groundedness": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.STRING),
    "intent_resolution": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "relevance": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "response_completeness": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "task_completion": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "task_adherence": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "tool_call_accuracy": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_input_accuracy": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_output_utilization": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_selection": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_call_success": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
}


def get_string_llm_output(score: int, explanation: str = DEFAULT_EXPLANATION) -> str:
    """
    Generate string-formatted LLM output.

    Args:
        score: The score value to include in the output
        explanation: The explanation text

    Returns:
        Formatted string output with score and explanation
    """
    return f"<S0>{explanation}</S0>" "<S1>Reasoning</S1>" f"<S2>{score}</S2>"


def get_dict_llm_output(score: int, explanation: str = DEFAULT_EXPLANATION) -> Dict[str, Any]:
    """
    Generate dictionary-formatted LLM output.

    Args:
        score: The score value to include in the output
        explanation: The explanation text

    Returns:
        Dictionary output with score and explanation
    """
    return {
        "llm_output": {
            "score": score,
            "label": "pass",
            "success": BINARY_SUCCESS_SCORE,
            "tool_calls_success_level": GRADERS_SUCCESS_SCORE,
            "result": score,
            "explanation": explanation,
            "reasoning": explanation,
        }
    }


def create_flow_side_effect(
    score: int, output_type: OutputType, explanation: str = DEFAULT_EXPLANATION
) -> Callable[[int], Awaitable[str | Dict[str, Any]]]:
    """
    Create async side effect functions for mocking evaluator flows.

    Args:
        score: The score to return in the mock output
        output_type: Whether to return string or dict format
        explanation: The explanation text to include

    Returns:
        Async function that can be used as a mock side effect
    """

    async def flow_side_effect(timeout, **kwargs):
        if output_type == OutputType.STRING:
            return get_string_llm_output(score, explanation)
        else:
            return get_dict_llm_output(score, explanation)

    return flow_side_effect


def get_flow_side_effect_for_evaluator(
    evaluator_name: str,
) -> Callable[[int], Awaitable[str | Dict[str, Any]]]:
    """
    Get the appropriate flow side effect function for a given evaluator.

    Args:
        evaluator_name: Name of the evaluator (e.g., "fluency", "relevance")

    Returns:
        Async function configured for the evaluator's output format

    Raises:
        ValueError: If evaluator_name is not recognized
    """
    if evaluator_name not in EVALUATOR_CONFIGS:
        raise ValueError(
            f"Evaluator '{evaluator_name}' not recognized. Available evaluators: {list(EVALUATOR_CONFIGS.keys())}"
        )

    config = EVALUATOR_CONFIGS[evaluator_name]
    return create_flow_side_effect(config.score, config.output_type)
