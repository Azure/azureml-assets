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
BINARY_INVERSE_SUCCESS_SCORE = 0  # For evaluators where lower is better (e.g., deflection_rate)
DEFAULT_EXPLANATION = "Success explanation"


class OutputType(Enum):
    """Types of outputs that evaluators can produce."""

    STRING = "string"  # XML format: <S0>...<S2>score</S2>
    SIMPLE_STRING = "simple_string"  # Just the score as a string (e.g., "5")
    DICT = "dict"


class EvaluatorCategory(Enum):
    """Categories of evaluators based on their scoring system."""

    GRADERS = "graders"  # Uses 1-5 scale
    BINARY = "binary"  # Uses 0-1 scale (higher is better)
    BINARY_INVERSE = "binary_inverse"  # Uses 0-1 scale (lower is better)


class EvaluatorOutputConfig:
    """Configuration for an evaluator's output format and scoring."""

    def __init__(self, category: EvaluatorCategory, output_type: OutputType):
        """Initialize configuration with category and output type."""
        self.category = category
        self.output_type = output_type
        if category == EvaluatorCategory.GRADERS:
            self.score = GRADERS_SUCCESS_SCORE
        elif category == EvaluatorCategory.BINARY_INVERSE:
            self.score = BINARY_INVERSE_SUCCESS_SCORE
        else:
            self.score = BINARY_SUCCESS_SCORE


# Mapping of evaluator names to their output configurations
EVALUATOR_CONFIGS: Dict[str, EvaluatorOutputConfig] = {
    "fluency": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "coherence": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "groundedness": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "similarity": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "intent_resolution": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "relevance": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "response_completeness": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "task_completion": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "task_adherence": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_call_accuracy": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "tool_input_accuracy": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_output_utilization": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_selection": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "tool_call_success": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "customer_satisfaction": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
    "deflection_rate": EvaluatorOutputConfig(EvaluatorCategory.BINARY_INVERSE, OutputType.DICT),
    "quality_grader": EvaluatorOutputConfig(EvaluatorCategory.BINARY, OutputType.DICT),
    "retrieval": EvaluatorOutputConfig(EvaluatorCategory.GRADERS, OutputType.DICT),
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
            "reason": explanation,
            "status": "completed",
            "properties": {
                "abstention": False,
                "relevance": GRADERS_SUCCESS_SCORE,
                "answerCompleteness": GRADERS_SUCCESS_SCORE,
                "queryType": "factual",
                "conversationIncomplete": False,
                "judgeConfidence": "high",
                "groundedness": GRADERS_SUCCESS_SCORE,
                "contextCoverage": GRADERS_SUCCESS_SCORE,
                "documentUtility": "high",
                "missingContextParts": [],
                "unsupportedClaims": [],
                "explanation": {},
            },
        }
    }


def get_dict_llm_output_none_score(reason: str = "Not applicable: intermediate response") -> Dict[str, Any]:
    """
    Generate dictionary-formatted LLM output with score=None (skipped/not_applicable).

    This simulates what the prompty returns when a response is not applicable
    for evaluation (e.g., intermediate tool-only responses).
    """
    return {
        "llm_output": {
            "score": None,
            "reason": reason,
            "status": "skipped",
            "properties": None,
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
        elif output_type == OutputType.SIMPLE_STRING:
            return str(score)
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


def create_none_score_flow_side_effect(
    reason: str = "Not applicable: intermediate response",
) -> Callable[[int], Awaitable[Dict[str, Any]]]:
    """
    Create async side effect that returns score=None (skipped/not_applicable).

    Used to test that evaluators handle None scores from _return_not_applicable_result
    without crashing on math.isnan(None).
    """

    async def flow_side_effect(timeout, **kwargs):
        return get_dict_llm_output_none_score(reason)

    return flow_side_effect


def create_mocked_evaluator(evaluator_cls, evaluator_name: str):
    """Create an evaluator instance with _flow mocked using standard side effect.

    Reusable factory for evaluators that only need model_config and a mocked _flow.
    For evaluators with extra init params (evaluation_level, multi_turn, etc.),
    use a custom helper instead.

    Args:
        evaluator_cls: The evaluator class to instantiate (e.g., FluencyEvaluator).
        evaluator_name: Name used to look up the mock config (e.g., "fluency").

    Returns:
        An evaluator instance with _flow replaced by a MagicMock.
    """
    import os
    from unittest.mock import MagicMock
    from azure.ai.evaluation import AzureOpenAIModelConfiguration

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = evaluator_cls(model_config=model_config)
    evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator(evaluator_name))
    return evaluator


def assert_none_score_result(result: Dict[str, Any], evaluator_name: str):
    """Assert that an evaluator result reflects a None/not_applicable score.

    Args:
        result: The evaluator output dict.
        evaluator_name: The evaluator key (e.g., "fluency", "groundedness").
    """
    assert result[evaluator_name] is None, (
        f"Expected {evaluator_name} score to be None, got {result[evaluator_name]}"
    )
    assert result[f"{evaluator_name}_result"] == "not_applicable", (
        f"Expected {evaluator_name}_result to be 'not_applicable', got {result.get(f'{evaluator_name}_result')}"
    )


# Intermediate response whose final assistant turn is an unresolved function_call
# (the agent has not yet produced a final answer). Shared by the intermediate-response
# not-applicable regression tests so every evaluator exercises the same input.
INTERMEDIATE_FUNCTION_CALL_RESPONSE = [
    {
        "run_id": "",
        "role": "assistant",
        "content": [
            {
                "type": "function_call",
                "tool_call_id": "call_15sVz7lMj1JbY4ea0Om8oigT",
                "name": "get_horoscope",
                "arguments": {"sign": "Aquarius"},
            }
        ],
    }
]


def build_none_score_evaluator(evaluator_cls, reason: str = "Not applicable: intermediate response"):
    """Build an evaluator whose prompty flows all return a None/skipped score.

    Every present flow attribute (``_flow``, ``_multi_turn_flow``, ``_groundedness_flow``)
    is replaced with a mock that yields ``score=None`` so both the turn-level and
    conversation-level paths exercise the not-applicable handling. Used by the shared
    skipped/intermediate regression helpers below.

    Args:
        evaluator_cls: The evaluator class to instantiate.
        reason: Reason text carried by the mocked skipped output.

    Returns:
        An evaluator instance with all flows mocked to return a None score.
    """
    import os
    from unittest.mock import MagicMock
    from azure.ai.evaluation import AzureOpenAIModelConfiguration

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = evaluator_cls(model_config=model_config)
    side_effect = create_none_score_flow_side_effect(reason)
    for attr in ("_flow", "_multi_turn_flow", "_groundedness_flow"):
        if hasattr(evaluator, attr):
            setattr(evaluator, attr, MagicMock(side_effect=side_effect))
    if hasattr(evaluator, "_ensure_query_prompty_loaded"):
        evaluator._ensure_query_prompty_loaded = MagicMock()
    return evaluator


def run_none_score_not_applicable(evaluator_cls, evaluator_name: str, **call_kwargs):
    """Run an evaluator whose flow returns score=None and assert a not-applicable result.

    Shared regression for the ``math.isnan(None)`` / skipped-status fix: when the flow
    yields ``score=None`` the evaluator must return the standardized not-applicable
    result instead of crashing. Callers pass their own valid inputs via ``call_kwargs``
    (e.g. ``query``/``response``/``context``/``ground_truth``/``messages``).

    Args:
        evaluator_cls: The evaluator class under test.
        evaluator_name: The evaluator result key (e.g. "fluency").
        **call_kwargs: Inputs forwarded to the evaluator call.

    Returns:
        The evaluator result dict (for optional extra assertions).
    """
    evaluator = build_none_score_evaluator(evaluator_cls)
    result = evaluator(**call_kwargs)
    assert_none_score_result(result, evaluator_name)
    return result


def run_intermediate_response_not_applicable(evaluator_cls, evaluator_name: str, response=None, **call_kwargs):
    """Run an evaluator with an intermediate (function_call) response; assert not-applicable.

    Regression: a response whose final assistant turn is an unresolved function_call
    must be treated as not-applicable (short-circuited before the LLM call) rather
    than evaluated.

    Args:
        evaluator_cls: The evaluator class under test.
        evaluator_name: The evaluator result key.
        response: Override the intermediate response (defaults to a single function_call).
        **call_kwargs: Additional inputs forwarded to the evaluator call.

    Returns:
        The evaluator result dict (for optional extra assertions).
    """
    evaluator = build_none_score_evaluator(evaluator_cls)
    result = evaluator(response=INTERMEDIATE_FUNCTION_CALL_RESPONSE if response is None else response, **call_kwargs)
    assert_none_score_result(result, evaluator_name)
    return result
