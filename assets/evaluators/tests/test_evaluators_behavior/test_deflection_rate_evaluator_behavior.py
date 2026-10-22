# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Deflection Rate Evaluator."""

import asyncio
import builtins
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation._exceptions import EvaluationException
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _MessagesUtilE2ETests, _TurnLevelUtilE2ETests
from .base_decision_routing_test import DecisionRoutingTests, construct_evaluator, enable_fake_decision_sdk
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
        """Messages input should route to the multi-turn flow."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        result = evaluator(messages=VALID_MESSAGES)

        assert result["deflection_rate"] == 0
        assert result["deflection_rate_status"] == "completed"
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_messages_are_serialized_for_multi_turn_prompt(self):
        """Messages should be serialized into the multi-turn prompt text."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        evaluator(messages=VALID_MESSAGES)

        conversation_text = evaluator._multi_turn_flow.call_args.kwargs["messages"]
        assert "Can you reset my password?" in conversation_text
        assert "Use Forgot Password on the sign-in page." in conversation_text

    def test_empty_messages_raises_error(self):
        """An empty messages list should raise an evaluation exception."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_non_list_messages_raises_error(self):
        """A non-list messages value should raise an evaluation exception."""
        evaluator = create_mocked_evaluator(DeflectionRateEvaluator, "deflection_rate")

        with pytest.raises(EvaluationException):
            evaluator(messages="not a list")

    def test_turn_evaluation_level_uses_response_flow_for_messages(self):
        """Turn-level evaluation should use the single-response flow."""
        evaluator = create_mocked_evaluator(
            DeflectionRateEvaluator,
            "deflection_rate",
            evaluation_level=EvaluationLevel.TURN,
        )

        evaluator(messages=VALID_MESSAGES)

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_conversation_evaluation_level_uses_multi_turn_flow_for_messages(self):
        """Forced conversation-level evaluation routes messages to the multi-turn flow."""
        evaluator = create_mocked_evaluator(
            DeflectionRateEvaluator,
            "deflection_rate",
            evaluation_level=EvaluationLevel.CONVERSATION,
        )

        evaluator(messages=VALID_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_conversation_evaluation_level_requires_messages(self):
        """Conversation-level evaluation without messages should raise an error."""
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


# region Native Decision routing coverage on the released SDK

DECISION_STRATEGY_MODULE = "azure.ai.evaluation._evaluators._common._decision_strategy"


@pytest.mark.unittest
class TestDeflectionRateDecisionRouting(DecisionRoutingTests):
    """Cover the native Decision branches of deflection_rate without a Decision-capable SDK."""

    result_key = "deflection_rate"
    class_name = "DeflectionRateEvaluator"
    inputs = {"response": "Use Forgot Password on the sign-in page."}
    scorings = ("_DeflectionScoring",)
    expected_strategies = 2

    def test_broken_decision_import_is_not_hidden(self, monkeypatch):
        """A broken SDK dependency propagates instead of being read as a missing Decision capability."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == DECISION_STRATEGY_MODULE:
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            self.load()
        assert caught.value is error

    def test_turn_state_is_the_response_and_the_category_is_surfaced(self, monkeypatch):
        """The turn-level native state is the prepared response, and the selected category becomes public."""
        module = self.load()
        verdict = {"deflection_rate": 1.0,
                   "deflection_rate_properties": {"decision_model": {"deflection_type": "reframe"}}}
        sdk = enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                       scorings=self.scorings, evaluate=verdict)
        evaluator, _ = construct_evaluator(module, self.class_name)
        result = evaluator(response="I can help with related topics instead.")
        assert result["deflection_rate"] == 1.0
        assert result["deflection_rate_deflection_type"] == "reframe"
        state = evaluator._strategy.evaluate.await_args.args[0]
        assert state == {"response": "I can help with related topics instead."}
        sdk.scorings["_DeflectionScoring"].assert_called_with(4, minimum=0)
        evaluator._multi_turn_strategy.evaluate.assert_not_awaited()

    def test_forced_turn_level_splits_messages_without_routing_keys(self, monkeypatch):
        """Forced turn evaluation of messages keeps only the latest agent response in the native state."""
        module = self.load()
        enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                 scorings=self.scorings)
        evaluator, _ = construct_evaluator(module, self.class_name, evaluation_level="turn")
        assert evaluator(messages=deepcopy(VALID_MESSAGES)) == {"deflection_rate": 1}
        state = evaluator._strategy.evaluate.await_args.args[0]
        assert set(state) == {"response"}
        assert "Use Forgot Password on the sign-in page." in state["response"]

    def test_conversation_state_is_the_serialized_transcript(self, monkeypatch):
        """Messages route to the conversation strategy with the serialized transcript as the only field."""
        module = self.load()
        enable_fake_decision_sdk(monkeypatch, module, self.class_name, result_key=self.result_key,
                                 scorings=self.scorings)
        evaluator, _ = construct_evaluator(module, self.class_name)
        assert evaluator(messages=deepcopy(VALID_MESSAGES)) == {"deflection_rate": 1}
        state = evaluator._multi_turn_strategy.evaluate.await_args.args[0]
        assert set(state) == {"messages"}
        assert "Can you reset my password?" in state["messages"]
        evaluator._strategy.evaluate.assert_not_awaited()

    def test_native_scoring_maps_categories_to_the_binary_metric(self, monkeypatch):
        """The category scoring maps plain_answer to 0 and every deflection category to 1, lower is better."""
        fake_sdk = types.ModuleType(DECISION_STRATEGY_MODULE)

        class FakeLikertScoring:
            """Minimal stand-in for the SDK scoring the deflection subclass extends."""

            def __init__(self, levels, minimum=1):
                self.levels = levels
                self.minimum = minimum

            def interpret(self, response, result_key, error_target):
                probabilities = response["answers"][result_key]["probabilities"]
                category = max(range(self.levels), key=lambda level: probabilities[str(level)]) + self.minimum
                expected = sum(level * probabilities[str(level)] for level in range(self.levels)) + self.minimum
                return float(category), "selected", {"probabilities": probabilities, "expected_score": expected}

            def passes(self, score, threshold):
                return score >= threshold

        fake_sdk.LikertScoring = FakeLikertScoring
        fake_sdk.DecisionEvaluationStrategy = object
        monkeypatch.setitem(sys.modules, DECISION_STRATEGY_MODULE, fake_sdk)
        module = self.load()
        scoring = module["_DeflectionScoring"](4, minimum=0)

        def answer(probabilities):
            return {"answers": {"deflection_rate": {"probabilities": probabilities}}}

        score, reason, details = scoring.interpret(answer({"0": 0.1, "1": 0.2, "2": 0.6, "3": 0.1}),
                                                   "deflection_rate", None)
        assert (score, details["deflection_type"], details["native_category"]) == (1.0, "send_elsewhere", 2)
        assert details["expected_score"] == pytest.approx(0.9)
        assert details["expected_category"] == pytest.approx(1.7)
        assert "send_elsewhere" in reason and "1" in reason
        score, _, details = scoring.interpret(answer({"0": 0.9, "1": 0.1, "2": 0.0, "3": 0.0}),
                                              "deflection_rate", None)
        assert (score, details["deflection_type"]) == (0.0, "plain_answer")
        assert scoring.passes(0.0, 0) and not scoring.passes(1.0, 0) and scoring.passes(1.0, 1)


# endregion
