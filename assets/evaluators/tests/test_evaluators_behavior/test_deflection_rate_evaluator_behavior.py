# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Deflection Rate Evaluator."""

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _MessagesUtilE2ETests, _TurnLevelUtilE2ETests
from ..common.evaluator_mock_config import create_mocked_evaluator, build_none_score_evaluator
from .base_validator_unit_test import (
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ...builtin.deflection_rate.evaluator._deflection_rate import (
    DeflectionRateEvaluator,
    EvaluationLevel,
)


@pytest.mark.unittest
class TestDeflectionRateEvaluatorBehavior(BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests, _MessagesUtilE2ETests):
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
            f"{self._result_prefix}_score",
            f"{self._result_prefix}_passed",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_status",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_properties",
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

        Deflection Rate uses the standardized not-applicable result shape.

        Args:
            result_data: Dictionary containing evaluation result data.

        Raises:
            AssertionError: If the result is not a valid not-applicable result
                for this evaluator.
        """
        label = result_data.get("label")
        reason = result_data.get("reason", "") or ""
        assert label == "not_applicable", f"Expected 'not_applicable' but got '{label}'"
        assert result_data.get("score") is None
        assert result_data.get("status") == "skipped"
        assert result_data.get("passed") is None
        assert "not applicable" in reason.lower(), \
            f"Expected reason to contain 'not applicable' but got '{reason}'"

    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result for Deflection Rate evaluator.

        Deflection Rate has an inverse threshold: lower scores pass.

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
        assert result_data.get("status") == "completed"
        assert score is not None, "Score should not be None"
        assert isinstance(score, (int, float)), \
            f"Score should be numeric but got type {type(score)}"
        assert score <= threshold, \
            f"Score {score} should be <= threshold {threshold}"


VALID_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "Can you reset my password?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Use Forgot Password on the sign-in page."}]},
]


@pytest.mark.unittest
class TestDeflectionRateMultiturnBehavior:
    """Behavioral tests for Deflection Rate conversation-level evaluation."""

    evaluator_type = DeflectionRateEvaluator

    def test_messages_uses_multi_turn_flow(self):
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        result = evaluator(messages=VALID_MESSAGES)

        assert result["deflection_rate"] == 0
        assert result["deflection_rate_status"] == "completed"
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_messages_are_serialized_for_multi_turn_prompt(self):
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        evaluator(messages=VALID_MESSAGES)

        conversation_text = evaluator._multi_turn_flow.call_args.kwargs["messages"]
        assert "Can you reset my password?" in conversation_text
        assert "Use Forgot Password on the sign-in page." in conversation_text

    def test_empty_messages_raises_error(self):
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_non_list_messages_raises_error(self):
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        with pytest.raises(EvaluationException):
            evaluator(messages="not a list")

    def test_turn_evaluation_level_uses_response_flow_for_messages(self):
        evaluator = create_mocked_evaluator(
            DeflectionRateEvaluator,
            "deflection_rate",
            evaluation_level=EvaluationLevel.TURN,
        )

        evaluator(messages=VALID_MESSAGES)

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_conversation_evaluation_level_requires_messages(self):
        evaluator = create_mocked_evaluator(
            DeflectionRateEvaluator,
            "deflection_rate",
            evaluation_level=EvaluationLevel.CONVERSATION,
        )

        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._real_call(response="A direct answer."))


# region Not-applicable handling tests (util-fix regression)

@pytest.mark.unittest
class TestDeflectionRateIntermediateResponse:
    """Regression test for the ``_is_intermediate_response`` rejection in ``_do_eval``.

    Deflection Rate's not-applicable result is bespoke: ``score=threshold`` and
    ``result='pass'`` with ``status='skipped'``, so the shared
    ``assert_none_score_result`` helper does not apply.
    """

    def test_intermediate_response_returns_not_applicable(self):
        """A trailing function_call response is treated as not-applicable (pass) before the LLM call."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")
        result = evaluator(response=BaseEvaluatorBehaviorTest.FUNCTION_CALL_ONLY_RESPONSE)
        assert result["deflection_rate_result"] == "not_applicable"
        assert result["deflection_rate"] is None
        assert result["deflection_rate_status"] == "skipped"
        assert "not applicable" in result["deflection_rate_reason"].lower()

    def test_skipped_llm_status_returns_not_applicable(self):
        """A skipped/None LLM score returns the standardized not-applicable result."""
        evaluator = build_none_score_evaluator(DeflectionRateEvaluator)
        result = evaluator(response="I'm sorry, I can't help with that. Please contact support.")
        assert result["deflection_rate"] is None
        assert result["deflection_rate_result"] == "not_applicable"
        assert result["deflection_rate_status"] == "skipped"


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
