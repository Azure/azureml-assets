# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Deflection Rate Evaluator."""

import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests
from ..common.evaluator_mock_config import create_mocked_evaluator, build_none_score_evaluator
from .base_validator_unit_test import (
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ...builtin.deflection_rate.evaluator._deflection_rate import (
    DeflectionRateEvaluator,
)


@pytest.mark.unittest
class TestDeflectionRateEvaluatorBehavior(BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests):
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


# region Not-applicable handling tests (util-fix regression)

@pytest.mark.unittest
class TestDeflectionRateIntermediateResponse:
    """Regression test for the ``_is_intermediate_response`` rejection in ``_do_eval``.

    Deflection Rate's not-applicable result is bespoke: ``score=threshold``,
    ``result='pass'`` and no ``status`` field, so the shared
    ``assert_none_score_result`` helper does not apply.
    """

    def test_intermediate_response_returns_not_applicable(self):
        """A trailing function_call response is treated as not-applicable (pass) before the LLM call."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")
        result = evaluator(response=BaseEvaluatorBehaviorTest.FUNCTION_CALL_ONLY_RESPONSE)
        assert result["deflection_rate_result"] == "pass"
        assert result["deflection_rate"] == result["deflection_rate_threshold"]
        assert "not applicable" in result["deflection_rate_reason"].lower()

    def test_skipped_llm_status_coerces_to_pass(self):
        """A skipped/None LLM score coerces to score=0 and a 'pass' result.

        Unlike the other evaluators, Deflection Rate has no ``status='skipped'``
        short-circuit, so a None score is coerced to 0 (deflected) rather than a
        not_applicable result. This is a known divergence tracked in
        EVALUATOR_DISCREPANCIES.md; the test pins the current behavior.
        """
        evaluator = build_none_score_evaluator(DeflectionRateEvaluator)
        result = evaluator(response="I'm sorry, I can't help with that. Please contact support.")
        assert result["deflection_rate"] == 0
        assert result["deflection_rate_result"] == "pass"


# endregion


@pytest.mark.unittest
class TestDeflectionRateValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ConversationValidatorUnitTests,
):
    """Low-level unit tests for deflection_rate's repeated validators, utils and methods."""

    evaluator_class = DeflectionRateEvaluator


# region _do_eval override branch coverage

@pytest.mark.unittest
class TestDeflectionRateDoEvalBranches:
    """Cover deflection_rate's override ``_do_eval`` raise and string-score branches."""

    def test_missing_response_raises(self):
        """A missing response raises a MISSING_FIELD error."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({}))

    def test_string_score_is_parsed(self):
        """A digit string score from the flow is parsed to an int."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        async def str_score_flow(timeout=None, **kwargs):
            return {"llm_output": {"score": "1", "explanation": "x", "deflection_type": "y"}}

        evaluator._flow = MagicMock(side_effect=str_score_flow)
        result = asyncio.run(evaluator._do_eval({"response": "The agent deflected the request."}))
        assert result["deflection_rate"] == 1

    def test_non_dict_output_raises(self):
        """A non-dict flow output raises an invalid-output error."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        async def str_flow(timeout=None, **kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = MagicMock(side_effect=str_flow)
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"response": "r"}))
