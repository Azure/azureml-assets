# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the Quality Evaluation Suite composite evaluator."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.quality_suite.evaluator._quality_suite import QualityEvaluationSuite, _EVALUATORS

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
    return QualityEvaluationSuite(model_config=model_config, **init_kwargs)


def _mock_flows(evaluator, llm_output):
    """Mock both prompty flows with the same LLM payload."""

    async def side_effect(timeout, **kwargs):
        return llm_output

    evaluator._flow = MagicMock(side_effect=side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=side_effect)
    return evaluator


@pytest.mark.unittest
class TestQualityEvaluationSuiteBehavior:
    """Behavioral tests for the QualityEvaluationSuite composite evaluator."""

    # region routing

    def test_query_response_uses_single_turn_flow(self):
        """Query/response input routes to the single-turn flow."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_response_only_uses_single_turn_flow(self):
        """A response-only input evaluates with an empty query."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(response=VALID_RESPONSE)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert evaluator._flow.call_args.kwargs["query"] == []

    def test_messages_uses_multi_turn_flow(self):
        """Message input routes to the multi-turn flow."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_conversation(self):
        """Conversation evaluation level routes query/response input to the multi-turn flow."""
        evaluator = _mock_flows(_make_evaluator(evaluation_level="conversation"), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_turn(self):
        """Turn evaluation level routes message input to the single-turn flow."""
        evaluator = _mock_flows(_make_evaluator(evaluation_level="turn"), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_turn_level_uses_assistant_only_messages_as_response(self):
        """Turn evaluation treats assistant-only messages as a response with an empty query."""
        evaluator = _mock_flows(_make_evaluator(evaluation_level="turn"), _all_completed_llm_output())
        assistant_messages = [
            {"role": "assistant", "content": [{"type": "text", "text": VALID_RESPONSE}]},
        ]
        evaluator(messages=assistant_messages)
        assert evaluator._flow.call_args.kwargs["query"] == []

    def test_invalid_evaluation_level_raises(self):
        """An invalid evaluation level raises an evaluation exception."""
        with pytest.raises(EvaluationException):
            _make_evaluator(evaluation_level="not_a_level")

    # endregion

    # region output shape and aggregation

    def test_primary_score_and_raw_evaluator_objects_are_nested(self):
        """Aggregate output keeps raw member results only in the nested evaluator map."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["quality_suite"] == 1
        assert result["quality_suite_threshold"] == 1
        assert result["quality_suite_result"] == "pass"
        assert result["quality_suite_passed"] is True
        evaluators = result["quality_suite_evaluators"]
        assert "evaluators" not in result["quality_suite_properties"]
        assert evaluators["fluency"]["threshold"] == 3
        assert evaluators["fluency"]["passed"] is True
        for name in _EVALUATOR_NAMES:
            assert name not in result
            assert evaluators[name]["status"] == "completed"

    def test_multi_turn_raw_failed_turn_is_preserved(self):
        """Multi-turn raw evaluator results preserve their failed-turn metadata."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output(failed_turn=1))
        result = evaluator(messages=VALID_MESSAGES)
        for name in _EVALUATOR_NAMES:
            assert result["quality_suite_evaluators"][name]["failed_turn"] == 1

    def test_default_thresholds_match_member_defaults(self):
        """Default aggregate thresholds match the member evaluator defaults."""
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
        """A partial threshold override preserves defaults for other members."""
        evaluator = _make_evaluator(threshold={"fluency": 4, "task_completion": 0})
        assert evaluator._threshold["fluency"] == 4
        assert evaluator._threshold["task_completion"] == 0
        assert evaluator._threshold["groundedness"] == 3

    def test_any_member_below_threshold_fails_primary_score(self):
        """A member below its threshold fails the aggregate result."""
        evaluator = _mock_flows(
            _make_evaluator(threshold={"fluency": 4}),
            _all_completed_llm_output(score_overrides={"fluency": 3}),
        )
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["quality_suite"] == 0
        assert result["quality_suite_threshold"] == 1
        assert result["quality_suite_result"] == "fail"
        assert result["quality_suite_passed"] is False
        assert result["quality_suite_evaluators"]["fluency"]["score"] == 3
        assert result["quality_suite_evaluators"]["fluency"]["threshold"] == 4
        assert result["quality_suite_evaluators"]["fluency"]["passed"] is False
        assert result["quality_suite_evaluators"]["coherence"]["score"] == 5

    # endregion

    # region skipped/not-applicable handling

    def test_all_members_skipped_returns_not_applicable(self):
        """An all-skipped member result returns an aggregate not-applicable result."""
        evaluator = _mock_flows(_make_evaluator(), _all_skipped_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["quality_suite"] is None
        assert result["quality_suite_threshold"] == 1
        assert result["quality_suite_result"] == "not_applicable"
        assert result["quality_suite_status"] == "skipped"
        assert result["quality_suite_passed"] is None
        for name in _EVALUATOR_NAMES:
            assert result["quality_suite_evaluators"][name]["score"] is None
            assert result["quality_suite_evaluators"][name]["threshold"] == next(
                evaluator["default_threshold"] for evaluator in _EVALUATORS if evaluator["name"] == name
            )
            assert result["quality_suite_evaluators"][name]["status"] == "skipped"
            assert result["quality_suite_evaluators"][name]["passed"] is None

    def test_intermediate_response_returns_not_applicable(self):
        """An intermediate function-call response is skipped without invoking the LLM."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        intermediate_response = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "tool_call_id": "call_1",
                        "name": "lookup_account",
                        "arguments": {"account_id": "123"},
                    }
                ],
            }
        ]
        result = evaluator(query=VALID_QUERY, response=intermediate_response)
        assert result["quality_suite"] is None
        assert result["quality_suite_status"] == "skipped"
        evaluator._flow.assert_not_called()

    def test_mixed_skipped_and_completed_members_pass(self):
        """Completed members pass when another member is skipped."""
        llm_output = _all_completed_llm_output()
        llm_output["llm_output"]["groundedness"] = {
            "score": None,
            "status": "skipped",
            "reason": "No factual claims.",
        }
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["quality_suite"] == 1
        assert result["quality_suite_evaluators"]["groundedness"]["status"] == "skipped"
        assert result["quality_suite_evaluators"]["fluency"]["status"] == "completed"

    def test_missing_member_output_is_treated_as_skipped(self):
        """An omitted member output is represented as a skipped evaluator."""
        llm_output = _all_completed_llm_output()
        del llm_output["llm_output"]["task_completion"]
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["quality_suite_evaluators"]["task_completion"]["score"] is None
        assert result["quality_suite_evaluators"]["task_completion"]["status"] == "skipped"
        assert result["quality_suite_evaluators"]["fluency"]["status"] == "completed"

    # endregion

    # region validation and malformed outputs

    def test_missing_response_raises(self):
        """A query without a response raises an evaluation exception."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY)

    def test_empty_messages_raises(self):
        """An empty messages list raises an evaluation exception."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_messages_missing_role_raises(self):
        """A message without a role raises an evaluation exception."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[{"content": "hi"}])

    def test_non_dict_llm_output_raises(self):
        """A non-dictionary LLM payload raises an evaluation exception."""
        evaluator = _mock_flows(_make_evaluator(), {"llm_output": "not a dict"})
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    def test_out_of_range_member_score_raises(self):
        """An out-of-range member score raises an evaluation exception."""
        evaluator = _mock_flows(
            _make_evaluator(),
            _all_completed_llm_output(score_overrides={"fluency": 6}),
        )
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    def test_boolean_member_score_raises(self):
        """A boolean member score raises an evaluation exception."""
        evaluator = _mock_flows(
            _make_evaluator(),
            _all_completed_llm_output(score_overrides={"fluency": True}),
        )
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    def test_do_eval_missing_response_raises(self):
        """The direct evaluation path requires a response."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": VALID_QUERY}))

    def test_do_eval_normalizes_missing_query_and_message_lists(self):
        """The direct evaluation path normalizes missing queries and message lists."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        asyncio.run(
            evaluator._do_eval(
                {
                    "query": None,
                    "response": [{"role": "assistant", "content": [{"type": "text", "text": VALID_RESPONSE}]}],
                }
            )
        )
        assert evaluator._flow.call_args.kwargs["query"] == []

    def test_super_real_call_handles_empty_multiple_and_conversion_errors(self):
        """The shared call path handles empty, multiple, and invalid converted inputs."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[])
        assert asyncio.run(evaluator._the_super_real_call()) == {}

        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"response": VALID_RESPONSE}] * 2)
        evaluator._do_eval = AsyncMock(return_value={"quality_suite": 1})
        evaluator._aggregate_results = MagicMock(return_value={"quality_suite": 1})
        assert asyncio.run(evaluator._the_super_real_call()) == {"quality_suite": 1}

        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("invalid input"))
        with pytest.raises(ValueError, match="invalid input"):
            asyncio.run(evaluator._the_super_real_call())

    # endregion
