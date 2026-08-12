# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the Tool Use Quality Evaluators meta-evaluator.

ToolUseQualityEvaluators batches five evaluators (tool_call_accuracy, tool_call_success,
tool_input_accuracy, tool_output_utilization, tool_selection) into a single LLM
call, so its ``llm_output`` shape (one JSON object keyed by evaluator name) differs
from the single-evaluator shape assumed by the shared ``BaseToolsEvaluatorBehaviorTest``
/ ``_TurnLevelUtilE2ETests`` / ``_MessagesUtilE2ETests`` infrastructure and by
``evaluator_mock_config.create_mocked_evaluator`` (which mocks a single-score
``llm_output``). This file therefore defines its own lightweight mocks tailored to
the multi-evaluator output shape instead of reusing that shared infrastructure.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.tool_use_quality.evaluator._tool_use_quality import ToolUseQualityEvaluators, _EVALUATORS

VALID_QUERY = "What's the weather in Seattle?"
VALID_RESPONSE = "The weather in Seattle is rainy at 14 degrees C."
VALID_TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Fetches the weather for a location.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
]

VALID_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": VALID_QUERY}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "arguments": {"city": "Seattle"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14C"}}],
    },
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


def _all_skipped_llm_output(reason="No tool calls were made."):
    output = {
        evaluator["name"]: {"score": None, "status": "skipped", "reason": reason}
        for evaluator in _EVALUATORS
    }
    return {"llm_output": output}


def _make_evaluator(**init_kwargs):
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    return ToolUseQualityEvaluators(model_config=model_config, **init_kwargs)


def _mock_flows(evaluator, llm_output):
    """Mock both ``_flow`` and ``_multi_turn_flow`` to return the same llm_output payload."""

    async def side_effect(timeout, **kwargs):
        return llm_output

    evaluator._flow = MagicMock(side_effect=side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=side_effect)
    return evaluator


@pytest.mark.unittest
class TestToolUseQualityEvaluatorsBehavior:
    """Behavioral tests for the ToolUseQualityEvaluators meta-evaluator."""

    # region routing

    def test_query_response_uses_single_turn_flow(self):
        """query/response input routes to the single-turn prompty flow."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_uses_multi_turn_flow(self):
        """messages input routes to the multi-turn prompty flow."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_conversation(self):
        """evaluation_level='conversation' forces the multi-turn path even for query/response input."""
        evaluator = _mock_flows(_make_evaluator(evaluation_level="conversation"), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_evaluation_level_forces_turn(self):
        """evaluation_level='turn' forces the single-turn path even for messages input."""
        evaluator = _mock_flows(_make_evaluator(evaluation_level="turn"), _all_completed_llm_output())
        evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_invalid_evaluation_level_raises(self):
        """An unrecognized evaluation_level value raises EvaluationException at construction time."""
        with pytest.raises(EvaluationException):
            _make_evaluator(evaluation_level="not_a_level")

    # endregion

    # region output shape

    def test_all_five_evaluators_are_nested_for_query_response(self):
        """The primary score and all five raw evaluator objects are present under the aggregate result."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] == 1
        assert result["tool_use_quality_result"] == "pass"
        assert result["tool_use_quality_passed"] is True
        evaluators = result["tool_use_quality_evaluators"]
        assert "evaluators" not in result["tool_use_quality_properties"]
        for name in _EVALUATOR_NAMES:
            assert name not in result
            assert evaluators[name]["score"] == next(
                evaluator["max"] for evaluator in _EVALUATORS if evaluator["name"] == name
            )
            assert evaluators[name]["status"] == "completed"

    def test_all_five_evaluators_present_messages(self):
        """All five evaluator result objects are nested for messages (multi-turn) input."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        for name in _EVALUATOR_NAMES:
            assert result["tool_use_quality_evaluators"][name] is not None
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["failed_turn"] is None

    def test_thresholds_default_to_standalone_evaluator_defaults(self):
        """Default thresholds match each standalone evaluator's default."""
        evaluator = _make_evaluator()
        assert evaluator._threshold["tool_call_accuracy"] == 3
        for name in ("tool_call_success", "tool_input_accuracy", "tool_output_utilization", "tool_selection"):
            assert evaluator._threshold[name] == 1

    def test_custom_threshold_partial_override(self):
        """A partial threshold dict only overrides the specified evaluator(s)."""
        evaluator = _make_evaluator(threshold={"tool_call_accuracy": 4})
        assert evaluator._threshold["tool_call_accuracy"] == 4
        assert evaluator._threshold["tool_selection"] == 1

    def test_tool_call_accuracy_fails_below_threshold(self):
        """An evaluator scoring below its threshold is marked failed independently of the others."""
        evaluator = _mock_flows(
            _make_evaluator(threshold={"tool_call_accuracy": 4}),
            _all_completed_llm_output(score_overrides={"tool_call_accuracy": 3}),
        )
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] == 0
        assert result["tool_use_quality_result"] == "fail"
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["score"] == 3
        assert result["tool_use_quality_evaluators"]["tool_selection"]["score"] == 1

    @pytest.mark.parametrize("score", [True, 6])
    def test_invalid_member_score_raises(self, score):
        evaluator = _mock_flows(
            _make_evaluator(),
            _all_completed_llm_output(score_overrides={"tool_call_accuracy": score}),
        )
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)

    # endregion

    # region skip / not-applicable handling

    def test_all_evaluators_skipped_when_llm_reports_skip(self):
        """When the LLM marks every evaluator skipped, all five results are not_applicable."""
        evaluator = _mock_flows(_make_evaluator(), _all_skipped_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] is None
        assert result["tool_use_quality_result"] == "not_applicable"
        for name in _EVALUATOR_NAMES:
            assert result["tool_use_quality_evaluators"][name]["score"] is None
            assert result["tool_use_quality_evaluators"][name]["status"] == "skipped"

    def test_mixed_skip_and_completed_evaluators(self):
        """One evaluator can be skipped while the others are completed, independently."""
        llm_output = _all_completed_llm_output()
        llm_output["llm_output"]["tool_output_utilization"] = {
            "score": None,
            "status": "skipped",
            "reason": "No tool outputs available.",
        }
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] == 1
        assert result["tool_use_quality_evaluators"]["tool_output_utilization"]["score"] is None
        assert result["tool_use_quality_evaluators"]["tool_output_utilization"]["status"] == "skipped"
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["status"] == "completed"
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["score"] == 5

    def test_intermediate_response_returns_not_applicable_for_all_evaluators(self):
        """An intermediate function-call-only response skips all evaluators without calling the LLM."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        intermediate_response = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "tool_call_id": "call_1",
                        "name": "get_weather",
                        "arguments": {"city": "Seattle"},
                    }
                ],
            }
        ]
        result = evaluator(query=VALID_QUERY, response=intermediate_response, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] is None
        for name in _EVALUATOR_NAMES:
            assert result["tool_use_quality_evaluators"][name]["score"] is None
            assert result["tool_use_quality_evaluators"][name]["status"] == "skipped"
        evaluator._flow.assert_not_called()

    # endregion

    # region validation

    def test_missing_response_raises(self):
        """Omitting response (with only query provided) raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, tool_definitions=VALID_TOOL_DEFINITIONS)

    def test_missing_query_raises(self):
        """Omitting query (with only response provided) raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)

    def test_tool_definitions_required_for_query_response(self):
        """Omitting tool_definitions raises because tool definitions are required for tool-use quality."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE)

    def test_tool_definitions_required_for_messages(self):
        """Omitting tool_definitions raises for multi-turn tool-use quality as well."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=VALID_MESSAGES)

    def test_tool_definitions_invalid_format_raises(self):
        """Malformed tool_definitions (missing required 'name') raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        invalid_tool_definitions = [
            {"description": "A tool", "parameters": {"type": "object", "properties": {}}},
        ]
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=invalid_tool_definitions)

    def test_empty_messages_raises(self):
        """An empty messages list raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[], tool_definitions=VALID_TOOL_DEFINITIONS)

    def test_messages_missing_role_raises(self):
        """A message missing the 'role' key raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            evaluator(messages=[{"content": "hi"}], tool_definitions=VALID_TOOL_DEFINITIONS)

    # endregion

    # region malformed LLM output

    def test_non_dict_llm_output_raises(self):
        """A non-dict llm_output payload raises EvaluationException."""
        evaluator = _mock_flows(_make_evaluator(), {"llm_output": "not a dict"})
        with pytest.raises(EvaluationException):
            evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)

    def test_missing_evaluator_in_llm_output_treated_as_not_applicable(self):
        """If the LLM omits an evaluator entirely, that evaluator alone falls back to not_applicable."""
        llm_output = _all_completed_llm_output()
        del llm_output["llm_output"]["tool_selection"]
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality_evaluators"]["tool_selection"]["score"] is None
        assert result["tool_use_quality_evaluators"]["tool_selection"]["status"] == "skipped"
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["status"] == "completed"

    def test_do_eval_missing_query_or_response_raises(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"response": VALID_RESPONSE}))

    def test_super_real_call_handles_empty_multiple_and_conversion_errors(self):
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[])
        assert asyncio.run(evaluator._the_super_real_call()) == {}

        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"response": VALID_RESPONSE}] * 2)
        evaluator._do_eval = AsyncMock(return_value={"tool_use_quality": 1})
        evaluator._aggregate_results = MagicMock(return_value={"tool_use_quality": 1})
        assert asyncio.run(evaluator._the_super_real_call()) == {"tool_use_quality": 1}

        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("invalid input"))
        with pytest.raises(ValueError, match="invalid input"):
            asyncio.run(evaluator._the_super_real_call())

    # endregion
