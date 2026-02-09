# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for code-based evaluator tests.

Supports deterministic evaluators that don't require LLM calls (e.g., BLEU, F1, ROUGE, METEOR, GLEU).
"""

from typing import Any, Dict, List

from .base_evaluator_runner import BaseEvaluatorRunner


class BaseCodeEvaluatorRunner(BaseEvaluatorRunner):
    """
    Base class for running code-based evaluators for testing.

    Code-based evaluators are deterministic and don't require LLM calls.
    They typically take simple string inputs (response, ground_truth) and return scores.

    Subclasses should implement:
    - evaluator_type: type[EvaluatorBase] - type of the evaluator (e.g., BleuScoreEvaluator)
    - result_key: str - the key for the score in results (e.g., "bleu_score", "f1_score")

    Subclasses may override:
    - result_prefix: str - the prefix for result/threshold keys (e.g., "bleu", "f1")
    - constructor_arg_names: list - names of constructor arguments to pass (default: ["threshold"])
    """

    # Subclasses may override
    constructor_arg_names = ["threshold"]

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for code evaluators."""
        return [f"{self._result_prefix}_score", f"{self._result_prefix}_result", f"{self._result_prefix}_threshold"]

    # ==================== CODE-SPECIFIC ASSERTION HELPERS ====================

    def assert_threshold_matches(self, result_data: Dict[str, Any], expected_threshold: float):
        """Assert that the threshold in results matches the expected value.

        Args:
            result_data: Dictionary containing evaluation result data.
            expected_threshold: Expected threshold value.

        Raises:
            AssertionError: If thresholds don't match.
        """
        assert result_data["threshold"] == expected_threshold, \
            f"Expected threshold {expected_threshold} but got {result_data['threshold']}"
