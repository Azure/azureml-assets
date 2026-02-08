# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for all evaluator tests.

Provides common interfaces and assertion helpers shared by both prompty-based and code-based evaluators.
"""

from abc import ABC
from typing import Any, Dict, List, Type
from unittest.mock import MagicMock
from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._exceptions import EvaluationException

from .evaluator_mock_config import EVALUATOR_CONFIGS, get_flow_side_effect_for_evaluator


class AbstractBaseEvaluatorRunner(ABC):
    """
    Abstract base class for running evaluators for testing.

    Provides common assertion helpers and interfaces that both prompty-based
    and code-based evaluators can use.

    Subclasses should implement:
    - evaluator_type: The type of evaluator to test
    - result_key: The key for the score in results

    Subclasses may override:
    - result_prefix: prefix for result/threshold keys
    - constructor_arg_names: kwargs routed to constructor (rest go to evaluator call)
    - use_mocking: if true, mock flow llm calls (only applicable for prompty evaluator)
    - _init_evaluator(): Create an evaluator instance
    - _run_evaluation(): Run the evaluator with inputs
    - _extract_and_print_result(): Extract and format results
    """

    # Subclasses must implement
    evaluator_type: Type[EvaluatorBase] = None
    result_key: str = None  # e.g., "bleu_score", "f1_score", "rouge_f1_score"

    # Subclasses may override
    result_prefix: str = None  # e.g., "bleu", "f1", "rouge"
    constructor_arg_names: List[str] = []
    use_mocking: bool = False

    @property
    def expected_result_fields(self) -> List[str]:
        return []

    @property
    def _result_prefix(self) -> str:
        """Get the result prefix, auto-deriving from result_key if not explicitly set."""
        if self.result_prefix is not None:
            return self.result_prefix
        if self.result_key is None:
            return None
        # Auto-derive: "bleu_score" -> "bleu", "f1_score" -> "f1"
        if self.result_key.endswith("_score"):
            return self.result_key[:-6]  # Strip "_score"
        return self.result_key

    def _init_evaluator(self, **kwargs) -> EvaluatorBase:
        """Create evaluator instance with specified threshold.

        Args:
            **kwargs: Keyword arguments passed to the evaluator constructor.

        Returns:
            Configured evaluator instance.

        Raises:
            ValueError: If evaluator_type, or result_key is not set.
        """
        if self.evaluator_type is None:
            raise ValueError("Evaluator type not set. Subclass must define evaluator_type.")
        if self.result_key is None:
            raise ValueError("Result key not set. Subclass must define result_key.")

        return self.evaluator_type(**kwargs)

    def _run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation and return results.

        Args:
            **kwargs: Keyword arguments. Args in constructor_arg_names are passed to
                      evaluator constructor, remaining kwargs are passed to the evaluator call.

        Returns:
            Dictionary containing evaluation results.
        """
        # Split kwargs into constructor args vs call args
        constructor_kwargs = {}
        call_kwargs = {}

        for key, value in kwargs.items():
            if key in self.constructor_arg_names:
                constructor_kwargs[key] = value
            else:
                call_kwargs[key] = value

        evaluator = self._init_evaluator(**constructor_kwargs)

        # Mock the flow only for behavioral tests
        if self.use_mocking:
            if hasattr(evaluator, "_flow"):
                evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator(self.result_key))

            # Special handling for groundedness evaluator to disable flow reloading
            if hasattr(evaluator, "_ensure_query_prompty_loaded"):
                evaluator._ensure_query_prompty_loaded = MagicMock()

        try:
            results = evaluator(**call_kwargs)
            return results
        except EvaluationException as e:
            print(f"Error during evaluation: {e}")
            return {
                f"{self.result_key}_error_message": e.message,
                f"{self.result_key}_error_code": e.category.name,
            }
        except Exception as e:
            print(f"Unexpected error during evaluation: {e}")
            return {
                f"{self.result_key}_error_message": str(e),
            }

    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Extract result fields and print them.

        Args:
            results: Raw evaluation results from the evaluator.
            test_label: Label for the test (used in print output).

        Returns:
            Dictionary with standardized result fields.
        """
        if f"{self.result_key}_error_message" not in results:
            for field in self.expected_result_fields:
                if field not in results:
                    raise ValueError(f"Expected result field '{field}' not found in results.")

        score = results.get(self.result_key)
        label = results.get(f"{self._result_prefix}_result")

        error_message = results.get(f"{self.result_key}_error_message")
        error_code = results.get(f"{self.result_key}_error_code")

        # Optional fields
        reason = results.get(f"{self.result_key}_reason")
        threshold = results.get(f"{self._result_prefix}_threshold")
        precision = results.get(f"{self._result_prefix}_precision")
        recall = results.get(f"{self._result_prefix}_recall")
        f1_score = results.get(f"{self._result_prefix}_f1_score")

        result = {
            "evaluator": self.result_key,
            "score": score,
            "label": label,
        }

        print(f"\nEvaluation Result for {self.result_key}:")
        print(f"\n[{test_label}] Score: {score}")
        print(f"  Result: {label}")
        if reason is not None:
            print(f"  Reason: {reason}")
            result["reason"] = reason
        if threshold is not None:
            print(f"  Threshold: {threshold}")
            result["threshold"] = threshold
        if precision is not None:
            print(f"  Precision: {precision}")
            result["precision"] = precision
        if recall is not None:
            print(f"  Recall: {recall}")
            result["recall"] = recall
        if f1_score is not None:
            print(f"  F1 Score: {f1_score}")
            result["f1_score"] = f1_score
        if error_message or error_code:
            print(f"  Error Message: {error_message}")
            print(f"  Error Type: {error_code}")
            result["error_message"] = error_message
            result["error_code"] = error_code

        return result

    # ==================== COMMON ASSERTION HELPERS ====================

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result does not meet passing criteria.
        """
        score_key = "score"
        label_key = "label"
        threshold = self._get_threshold(result_data)
        assert result_data[label_key] == "pass", f"Expected 'pass' but got '{result_data[label_key]}'"
        assert result_data[score_key] is not None, "Score should not be None"
        score = result_data[score_key]
        score_type = type(score)
        assert score_type in [int, float], f"Score should be numeric but got type {score_type}"
        assert result_data[score_key] >= threshold, \
            f"Score {result_data[score_key]} should be >= threshold {threshold}"

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result does not meet failing criteria.
        """
        score_key = "score"
        label_key = "label"
        threshold = self._get_threshold(result_data)
        assert result_data[label_key] == "fail", f"Expected 'fail' but got '{result_data[label_key]}'"
        assert result_data[score_key] is not None, "Score should not be None"
        score = result_data[score_key]
        score_type = type(score)
        assert score_type in [int, float], f"Score should be numeric but got type {score_type}"
        assert result_data[score_key] < threshold, \
            f"Score {result_data[score_key]} should be < threshold {threshold}"

    def assert_pass_or_fail(self, result_data: Dict[str, Any]):
        """Assert a valid pass or fail result.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result is not a valid pass/fail.
        """
        score_key = "score"
        label_key = "label"
        assert result_data[label_key] in ["pass", "fail"], \
            f"Expected 'pass' or 'fail' but got '{result_data[label_key]}'"
        assert result_data[score_key] is not None, "Score should not be None"
        score = result_data[score_key]
        score_type = type(score)
        assert score_type in [int, float], f"Score should be numeric but got type {score_type}"
        assert score >= 0.0, f"Score {score} should be >= 0.0"

    def assert_score_in_range(self, result_data: Dict[str, Any], min_score: float = 0.0, max_score: float = 1.0):
        """Assert that score is within expected range.

        Args:
            result_data: Dictionary containing evaluation result data.
            min_score: Minimum expected score (inclusive).
            max_score: Maximum expected score (inclusive).

        Raises:
            AssertionError: If the score is outside the expected range.
        """
        score_key = "score"
        assert result_data[score_key] is not None, "Score should not be None"
        score = result_data[score_key]
        score_type = type(score)
        assert score_type in [int, float], f"Score should be numeric but got type {score_type}"
        assert min_score <= result_data[score_key] <= max_score, \
            f"Score {result_data[score_key]} should be in range [{min_score}, {max_score}]"

    def assert_error(self, result_data: Dict[str, Any], error_code: str = None):
        """Assert an error result.

        Args:
            result_data: Dictionary containing evaluation result data.
            error_code: Expected error code.

        Raises:
            AssertionError: If no error is present or error type doesn't match.
        """
        assert result_data.get("error_message") is not None, "Expected an error message"
        if error_code is not None:
            assert result_data["error_code"] == error_code, \
                f"Expected error type/code '{error_code}' but got '{result_data['error_code']}'"

    # ==================== HELPER METHODS ====================

    def _get_threshold(self, result_data: Dict[str, Any]) -> float:
        """Get the threshold for pass/fail determination.

        Args:
            result_data: Dictionary containing evaluation result data.

        Returns:
            Threshold value for pass/fail determination.
        """
        # Default: try to get from result_data, fallback to 0.5
        if "threshold" in result_data:
            return result_data["threshold"]

        evaluator_name = result_data["evaluator"]
        if evaluator_name in EVALUATOR_CONFIGS:
            grader_score = EVALUATOR_CONFIGS[evaluator_name].score
            threshold = float(grader_score) / 2.0
            return threshold

        return 0.5  # Default threshold if not provided
