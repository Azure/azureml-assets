# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Deflection Rate Evaluator."""

from typing import Any, Dict

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.deflection_rate.evaluator._deflection_rate import (
    DeflectionRateEvaluator,
)


@pytest.mark.unittest
class TestDeflectionRateEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Deflection Rate Evaluator.

    Tests different input formats and scenarios.
    Note: This evaluator only requires response, not query.
    """

    evaluator_type = DeflectionRateEvaluator

    # Deflection rate only needs response, not query
    requires_query = False

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.MINIMAL_RESPONSE

    @property
    def expected_result_fields(self):
        """Get the expected result fields for deflection rate evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_deflection_type",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]

    def assert_not_applicable(self, result_data: Dict[str, Any]):
        """Assert a not-applicable result for Deflection Rate evaluator.

        The Deflection Rate evaluator's not-applicable result currently uses
        ``label='pass'``, ``score=threshold`` (e.g. 0), and does not emit
        ``passed`` or ``status`` fields. The reason still begins with
        ``"Not applicable"``. This override matches that behavior.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result is not a valid not-applicable result
                for this evaluator.
        """
        label = result_data.get("label")
        reason = result_data.get("reason", "") or ""
        assert label == "pass", f"Expected 'pass' but got '{label}'"
        assert "not applicable" in reason.lower(), \
            f"Expected reason to contain 'not applicable' but got '{reason}'"

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result for Deflection Rate evaluator.

        The Deflection Rate evaluator does not emit ``passed`` or ``status``
        fields in its result dict; it only emits ``label`` (``"pass"``),
        ``score``, ``threshold``, and ``reason``. This override relaxes the
        ``passed is True`` / ``status == "completed"`` checks while still
        validating ``label == "pass"`` and that the score is numeric and
        meets the threshold.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result is not a valid pass result for this
                evaluator.
        """
        threshold = self._get_threshold(result_data)
        label = result_data.get("label")
        score = result_data.get("score")
        assert label == "pass", f"Expected 'pass' but got '{label}'"
        assert score is not None, "Score should not be None"
        assert isinstance(score, (int, float)), \
            f"Score should be numeric but got type {type(score)}"
        assert score >= threshold, \
            f"Score {score} should be >= threshold {threshold}"
