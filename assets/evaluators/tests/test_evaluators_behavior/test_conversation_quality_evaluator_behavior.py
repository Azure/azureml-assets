# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the Conversation Quality Evaluators meta-evaluator."""

import os
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.conversation_quality.evaluator._conversation_quality import ConversationQualityEvaluators, _EVALUATORS

VALID_QUERY = "How do I reset my account password?"
VALID_RESPONSE = "Open account settings, select Security, and choose Reset password."
VALID_TOOL_DEFINITIONS = [
    {
        "name": "lookup_account",
        "description": "Looks up account information.",
        "parameters": {"type": "object", "properties": {"account_id": {"type": "string"}}},
    }
]
VALID_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": VALID_QUERY}]},
    {"role": "assistant", "content": [{"type": "text", "text": VALID_RESPONSE}]},
]
_EVALUATOR_NAMES = [evaluator["name"] for evaluator in _EVALUATORS]


def _all_completed_llm_output(score_overrides=None, failed_turn=None):
    """Build an llm_output dict where every evaluator is completed with a passing score."""
    score_overrides = score_overrides or {}
    output = {}
    for evaluator in _EVALUATORS:
        name = evaluator["name"]
        score = score_overrides.get(name, evaluator["max"])
        output[name] = {
            "score": score,
            "status": "completed",
            "reason": f"{name} reason",
            "failed_turn": failed_turn,
        }
    return {"llm_output": output}


def _all_skipped_llm_output(reason="Not applicable."):
    return {
        "llm_output": {
            evaluator["name"]: {"score": None, "status": "skipped", "reason": reason}
            for evaluator in _EVALUATORS
        }
    }


def _make_evaluator(**init_kwargs):
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    return ConversationQualityEvaluators(model_config=model_config, **init_kwargs)


def _mock_flows(evaluator, llm_output):
    """Mock both prompty flows with the same LLM payload."""

    async def side_effect(timeout, **kwargs):
        return llm_output

    evaluator._flow = MagicMock(side_effect=side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=side_effect)
    return evaluator


@pytest.mark.unittest
class TestConversationQualityEvaluatorsBehavior:
    """Behavioral tests for the ConversationQualityEvaluators meta-evaluator."""

    # region routing

    def test_query_response_uses_single_turn_flow(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_response_only_uses_single_turn_flow(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(response=VALID_RESPONSE)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert evaluator._flow.call_args.kwargs["query"] == []

    def test_messages_uses_multi_turn_flow(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_conversation(self):
        evaluator = _mock_flows(_make_evaluator(evaluation_level="conversation"), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_turn(self):
        evaluator = _mock_flows(_make_evaluator(evaluation_level="turn"), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_invalid_evaluation_level_raises(self):
        with pytest.raises(EvaluationException):
            _make_evaluator(evaluation_level="not_a_level")

    # endregion

    # region output shape and aggregation

    def test_primary_score_and_raw_evaluator_objects_are_present(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["conversation_quality"] == 1
        assert result["conversation_quality_result"] == "pass"
        assert result["conversation_quality_passed"] is True
        assert "conversation_quality_evaluators" in result
        for name in _EVALUATOR_NAMES:
            assert result[name]["status"] == "completed"
            assert result["conversation_quality_evaluators"][name] is result[name]

    def test_multi_turn_raw_failed_turn_is_preserved(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output(failed_turn=1))
        result = evaluator(messages=VALID_MESSAGES)
        for name in _EVALUATOR_NAMES:
            assert result[name]["failed_turn"] == 1

    def test_default_thresholds_match_member_defaults(self):
        evaluator = _make_evaluator()
        assert evaluator._threshold == {
            "fluency": 3,
            "coherence": 3,
            "intent_resolution": 3,
            "task_adherence": 1,
            "groundedness": 3,
            "task_completion": 1,
        }

    def test_custom_threshold_partial_override(self):
        evaluator = _make_evaluator(threshold={"fluency": 4, "task_completion": 0})
        assert evaluator._threshold["fluency"] == 4
        assert evaluator._threshold["task_completion"] == 0
        assert evaluator._threshold["groundedness"] == 3

    def test_any_member_below_threshold_fails_primary_score(self):
        evaluator = _mock_flows(
            _make_evaluator(threshold={"fluency": 4}),
            _all_completed_llm_output(score_overrides={"fluency": 3}),
        )
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["conversation_quality"] == 0
        assert result["conversation_quality_result"] == "fail"
        assert result["conversation_quality_passed"] is False
        assert result["fluency"]["score"] == 3
        assert result["coherence"]["score"] == 5

    # endregion

    # region skipped/not-applicable handling

    def test_all_members_skipped_returns_not_applicable(self):
        evaluator = _mock_flows(_make_evaluator(), _all_skipped_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["conversation_quality"] is None
        assert result["conversation_quality_result"] == "not_applicable"
        assert result["conversation_quality_status"] == "skipped"
        for name in _EVALUATOR_NAMES:
            assert result[name]["score"] is None
            assert result[name]["status"] == "skipped"

    def test_mixed_skipped_and_completed_members_pass(self):
        llm_output = _all_completed_llm_output()
        llm_output["llm_output"]["groundedness"] = {
            "score": None,
            "status": "skipped",
            "reason": "No factual claims.",
        }
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["conversation_quality"] == 1
        assert result["groundedness"]["status"] == "skipped"
        assert result["fluency"]["status"] == "completed"

    def test_missing_member_output_is_treated_as_skipped(self):
        llm_output = _all_completed_llm_output()
        del llm_output["llm_output"]["task_completion"]
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["task_completion"]["score"] is None
        assert result["task_completion"]["status"] == "skipped"
        assert result["fluency"]["status"] == "completed"

    # endregion

    # region validation and malformed outputs

    def test_missing_response_raises(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY)

    def test_empty_messages_raises(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_messages_missing_role_raises(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[{"content": "hi"}])

    def test_non_dict_llm_output_raises(self):
        evaluator = _mock_flows(_make_evaluator(), {"llm_output": "not a dict"})
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    def test_out_of_range_member_score_raises(self):
        evaluator = _mock_flows(
            _make_evaluator(),
            _all_completed_llm_output(score_overrides={"fluency": 6}),
        )
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    # endregion
