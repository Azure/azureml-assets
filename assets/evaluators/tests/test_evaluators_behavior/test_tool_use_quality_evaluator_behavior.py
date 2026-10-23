# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the Tool Use Evaluation Suite composite evaluator.

ToolUseQualityEvaluator batches five evaluators (tool_call_accuracy, tool_call_success,
tool_input_accuracy, tool_output_utilization, tool_selection) into a single LLM
call, so its ``llm_output`` shape (one JSON object keyed by evaluator name) differs
from the single-evaluator shape assumed by the shared ``BaseToolsEvaluatorBehaviorTest``
/ ``_TurnLevelUtilE2ETests`` / ``_MessagesUtilE2ETests`` infrastructure and by
``evaluator_mock_config.create_mocked_evaluator`` (which mocks a single-score
``llm_output``). This file therefore defines its own lightweight mocks tailored to
the multi-evaluator output shape instead of reusing that shared infrastructure.
"""

import asyncio
import json
from pathlib import Path
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jsonschema
import pytest
import yaml
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.tool_use_quality.evaluator import _tool_use_quality as tool_use_quality_module
from ...builtin.tool_use_quality.evaluator._tool_use_quality import ToolUseQualityEvaluator, _EVALUATORS

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


def _llm_output_without_tool_definitions():
    """Build output where only definition-dependent member evaluators are skipped."""
    output = _all_completed_llm_output()["llm_output"]
    for name in ("tool_call_accuracy", "tool_input_accuracy", "tool_selection"):
        output[name] = {
            "score": None,
            "status": "skipped",
            "reason": "Tool definitions are missing.",
            "failed_turn": None,
        }
    return {"llm_output": output}


def _make_evaluator(**init_kwargs):
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    return ToolUseQualityEvaluator(model_config=model_config, **init_kwargs)


def _mock_flows(evaluator, llm_output):
    """Mock both ``_flow`` and ``_multi_turn_flow`` to return the same llm_output payload."""

    async def side_effect(timeout, **kwargs):
        return llm_output

    evaluator._flow = MagicMock(side_effect=side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=side_effect)
    return evaluator


@pytest.fixture
def mocked_decision_evaluator(monkeypatch):
    """Initialize chat internally while mocking only this evaluator's Decision contracts."""
    namespace = tool_use_quality_module.__dict__
    base = tool_use_quality_module.ModelEvaluatorBase
    original_init = base.__init__
    runner = AsyncMock(probability_threshold=0.8)
    configurations = []

    def initialize_chat(instance, **kwargs):
        configurations.append(kwargs["model_config"])
        kwargs["model_config"] = {
            "azure_endpoint": "https://example.openai.azure.com",
            "azure_deployment": "test-model",
        }
        original_init(instance, **kwargs)

    def make_strategy(selected_runner, **kwargs):
        name = kwargs["result_key"]

        def parse(response, eval_input):
            answer = response["answers"][name]
            status = response["answers"]["status"]
            return {
                f"{name}_score": answer["score"] if status == "completed" else None,
                f"{name}_status": status,
                f"{name}_reason": f"{name} reason",
                f"{name}_properties": answer.get("properties"),
            }

        return Mock(questions=kwargs["questions"], parse=Mock(side_effect=parse))

    monkeypatch.setattr(base, "__init__", initialize_chat)
    monkeypatch.setattr(base, "_decision_model_runner", property(lambda instance: runner), raising=False)
    monkeypatch.setitem(namespace, "_SDK_SUPPORTS_DECISION", True)
    binary_scoring = Mock()
    likert_scoring = Mock()
    strategy_factory = Mock(side_effect=make_strategy)
    monkeypatch.setitem(namespace, "BinaryScoring", binary_scoring)
    monkeypatch.setitem(namespace, "LikertScoring", likert_scoring)
    monkeypatch.setitem(namespace, "DecisionEvaluationStrategy", strategy_factory)
    from azure.ai.evaluation._legacy.prompty import _yaml_utils

    rubric_loader = Mock(wraps=_yaml_utils.load_yaml)
    monkeypatch.setattr(_yaml_utils, "load_yaml", rubric_loader)
    config = {
        "type": "decision",
        "azure_endpoint": "https://example.services.ai.azure.com",
        "azure_deployment": "Decision-1",
    }
    with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load", return_value=AsyncMock()) as load:
        evaluator = ToolUseQualityEvaluator(
            config, probability_threshold=0.8, threshold={"tool_call_accuracy": 4}
        )
    return SimpleNamespace(
        evaluator=evaluator, runner=runner, load=load, config=config, configurations=configurations,
        binary_scoring=binary_scoring, likert_scoring=likert_scoring,
        strategy_factory=strategy_factory, rubric_loader=rubric_loader,
    )


@pytest.mark.unittest
class TestToolUseQualityDecisionBehavior:
    """Exercise the composite Decision branch without importing SDK strategy modules."""

    def test_constructor_loads_paired_rubrics_without_multi_turn_chat(self, mocked_decision_evaluator):
        """Build ten member strategies from both rubrics without loading multi-turn chat."""
        mocks = mocked_decision_evaluator
        evaluator = mocks.evaluator
        assert mocks.configurations == [{**mocks.config, "probability_threshold": 0.8}]
        assert "probability_threshold" not in mocks.config
        mocks.load.assert_called_once()
        assert not hasattr(evaluator, "_multi_turn_flow")
        assert mocks.rubric_loader.call_count == 2
        assert [os.path.basename(call.args[0]) for call in mocks.rubric_loader.call_args_list] == [
            evaluator._DECISION_FILE, evaluator._MULTI_TURN_DECISION_FILE,
        ]
        assert mocks.strategy_factory.call_count == 10
        assert mocks.binary_scoring.call_count == 8
        for call in mocks.binary_scoring.call_args_list:
            assert call.args == (0.8,)
        assert mocks.likert_scoring.call_count == 2
        for call in mocks.likert_scoring.call_args_list:
            assert call.args == (5,)
            assert call.kwargs == {"minimum": 1}
        for call in mocks.strategy_factory.call_args_list:
            name = call.kwargs["result_key"]
            assert call.args == (mocks.runner,)
            assert set(call.kwargs["questions"]) == {"status", name}
            assert call.kwargs["questions"]["status"]["type"] == "noul"
            assert call.kwargs["questions"][name]["type"] == ("score" if name == "tool_call_accuracy" else "noul")
            assert call.kwargs["threshold"] == (4 if name == "tool_call_accuracy" else 1)
            assert call.kwargs["timeout"] == evaluator._LLM_CALL_TIMEOUT
            expected_scoring = mocks.likert_scoring if name == "tool_call_accuracy" else mocks.binary_scoring
            assert call.kwargs["scoring"] is expected_scoring.return_value

    @pytest.mark.parametrize("conversation", [False, True], ids=["turn", "conversation"])
    @pytest.mark.parametrize("outcome", ["pass", "mixed_pass", "mixed_fail", "skipped"])
    def test_one_request_maps_members_independently(self, mocked_decision_evaluator, conversation, outcome):
        """Route one request and preserve independent member statuses, scores, and shared usage."""
        mocks = mocked_decision_evaluator
        evaluator = mocks.evaluator
        answers = {}
        for member in _EVALUATORS:
            name = member["name"]
            skipped = outcome == "skipped" or (outcome.startswith("mixed") and name == "tool_selection")
            answers[f"{name}_status"] = "skipped" if skipped else "completed"
            answers[name] = {"score": member["max"]}
        if outcome == "mixed_fail":
            answers["tool_call_accuracy"]["score"] = 3
        native = {"native_answer": {"noul": 0.9}, "rubric_hash": "fixture"}
        answers["tool_call_success"]["properties"] = {"decision_model": native, "total_tokens": 999}
        response = {"answers": answers, "usage": {"input_tokens": 11, "output_tokens": 7}, "model": "Decision-1"}
        mocks.runner.return_value = response
        inputs = {"messages": VALID_MESSAGES} if conversation else {
            "query": VALID_MESSAGES[:1], "response": VALID_MESSAGES[1:],
        }
        result = evaluator(**inputs)
        mocks.runner.assert_awaited_once()
        request = mocks.runner.await_args.kwargs
        selected = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
        unused = evaluator._decision_strategies if conversation else evaluator._multi_turn_decision_strategies
        expected_questions = {}
        for name, strategy in selected.items():
            expected_questions[f"{name}_status"] = strategy.questions["status"]
            expected_questions[name] = strategy.questions[name]
            strategy.parse.assert_called_once()
            parsed_response, state = strategy.parse.call_args.args
            assert state == request["state"]
            assert parsed_response == {
                **response, "answers": {"status": answers[f"{name}_status"], name: answers[name]},
            }
            unused[name].parse.assert_not_called()
            assert result[f"{name}_status"] == answers[f"{name}_status"]
            expected_score = answers[name]["score"] if answers[f"{name}_status"] == "completed" else None
            assert result[f"{name}_score"] == expected_score
            assert result[f"{name}_reason"] == f"{name} reason"
            expected_result = "not_applicable" if expected_score is None else (
                "pass" if expected_score >= evaluator._threshold_for(name) else "fail"
            )
            assert result[f"{name}_result"] == expected_result
        assert request["questions"] == expected_questions
        assert len(request["questions"]) == 10
        assert request["timeout"] == evaluator._LLM_CALL_TIMEOUT
        if conversation:
            assert "User turn 1:" in request["state"]["messages"]
            assert request["state"]["tool_definitions"] == "TOOL_DEFINITIONS:"
        else:
            assert set(request["state"]) == {"query", "response", "tool_definitions"}
            assert request["state"]["tool_definitions"] == []
        expected_aggregate = None if outcome == "skipped" else 0 if outcome == "mixed_fail" else 1
        assert result["tool_use_quality"] == expected_aggregate
        assert result["tool_use_quality_result"] == (
            "not_applicable" if outcome == "skipped" else "fail" if outcome == "mixed_fail" else "pass"
        )
        assert result["tool_use_quality_status"] == ("skipped" if outcome == "skipped" else "completed")
        assert result["tool_use_quality_properties"]["failed_evaluators"] == (
            ["tool_call_accuracy"] if outcome == "mixed_fail" else []
        )
        assert result["tool_use_quality_properties"]["skipped_evaluators"] == (
            _EVALUATOR_NAMES if outcome == "skipped" else ["tool_selection"] if outcome.startswith("mixed") else []
        )
        assert result["tool_use_quality_evaluators"]["tool_call_success"]["properties"] == {"decision_model": native}
        assert result["tool_use_quality_evaluators"]["tool_selection"]["properties"] == {}
        assert result["tool_use_quality_prompt_tokens"] == 11
        assert result["tool_use_quality_completion_tokens"] == 7
        assert result["tool_use_quality_total_tokens"] == 18
        assert result["tool_use_quality_model"] == "Decision-1"

    def test_older_sdk_rejects_decision_before_base_initialization(self, monkeypatch):
        """Reject a Decision configuration when the installed SDK lacks parse support."""
        monkeypatch.setattr(tool_use_quality_module, "_SDK_SUPPORTS_DECISION", False)
        with patch.object(tool_use_quality_module.ModelEvaluatorBase, "__init__") as initialize:
            with pytest.raises(EvaluationException, match="Decision strategy parse support"):
                ToolUseQualityEvaluator({"type": "decision"})
        initialize.assert_not_called()

    def test_chat_rejects_decision_probability_threshold(self):
        """Reject the Decision-only cutoff before loading a chat model."""
        with patch("azure.ai.evaluation._legacy.prompty.AsyncPrompty.load") as load:
            with pytest.raises(EvaluationException, match="only supported for Decision models"):
                _make_evaluator(probability_threshold=0.8)
        load.assert_not_called()


@pytest.mark.unittest
class TestToolUseQualityEvaluatorBehavior:
    """Behavioral tests for the ToolUseQualityEvaluator."""

    # region routing

    def test_query_response_uses_single_turn_flow(self):
        """query/response input routes to the single-turn prompty flow."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_uses_multi_turn_flow(self):
        """Messages input routes to the multi-turn prompty flow."""
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

    def test_all_five_evaluators_are_nested_and_flattened_for_query_response(self):
        """Member objects remain nested and expose converter-compatible flat keys."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=VALID_TOOL_DEFINITIONS)
        assert result["tool_use_quality"] == 1
        assert result["tool_use_quality_threshold"] == 1
        assert result["tool_use_quality_result"] == "pass"
        assert result["tool_use_quality_passed"] is True
        evaluators = result["tool_use_quality_evaluators"]
        assert "evaluators" not in result["tool_use_quality_properties"]
        assert evaluators["tool_call_accuracy"]["threshold"] == 3
        assert evaluators["tool_call_accuracy"]["passed"] is True
        for name in _EVALUATOR_NAMES:
            assert name not in result
            assert evaluators[name]["score"] == next(
                evaluator["max"] for evaluator in _EVALUATORS if evaluator["name"] == name
            )
            assert evaluators[name]["status"] == "completed"
            for field in ("score", "reason", "threshold", "status"):
                assert result[f"{name}_{field}"] == evaluators[name][field]
            assert result[f"{name}_result"] == "pass"
            assert f"{name}_passed" not in result

    def test_all_five_evaluators_present_messages(self):
        """All five evaluator result objects are nested for messages (multi-turn) input."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)
        for name in _EVALUATOR_NAMES:
            assert result["tool_use_quality_evaluators"][name] is not None
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["failed_turn"] is None

    @pytest.mark.parametrize("input_kwargs", [
        {
            "query": VALID_QUERY,
            "response": VALID_RESPONSE,
            "tool_definitions": VALID_TOOL_DEFINITIONS,
        },
        {"messages": VALID_MESSAGES, "tool_definitions": VALID_TOOL_DEFINITIONS},
    ])
    def test_cached_tokens_are_exposed_for_both_flow_types(self, input_kwargs):
        """Cached prompt tokens are available to the SDK converter for either Prompty flow."""
        llm_output = _all_completed_llm_output()
        llm_output["cached_tokens"] = 128
        evaluator = _mock_flows(_make_evaluator(), llm_output)

        result = evaluator(**input_kwargs)

        assert result["tool_use_quality_cached_tokens"] == 128
        assert result["tool_use_quality_properties"]["cached_tokens"] == 128
        assert (
            result["tool_use_quality_properties"][
                tool_use_quality_module.PROMPT_CACHE_CACHED_TOKENS_PROPERTY
            ]
            == 128
        )

    def test_cached_token_capture_helpers_are_self_contained(self, monkeypatch):
        """The evaluator file captures cached tokens without package-local imports."""

        class Details:
            cached_tokens = 29

        class Usage:
            prompt_tokens_details = Details()

        class Response:
            usage = Usage()

        assert tool_use_quality_module._extract_cached_tokens(
            {"usage": {"prompt_tokens_details": {"cached_tokens": 27}}}
        ) == 27
        assert tool_use_quality_module._extract_cached_tokens(Response()) == 29
        assert tool_use_quality_module._extract_cached_tokens({}) is None
        assert tool_use_quality_module._extract_cached_tokens({"usage": {}}) is None
        assert tool_use_quality_module._extract_cached_tokens(
            {"usage": {"prompt_tokens_details": {"cached_tokens": True}}}
        ) is None

        from openai.resources.chat.completions import AsyncCompletions

        async def fake_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 31}}}

        monkeypatch.setattr(AsyncCompletions, "create", fake_create)
        tool_use_quality_module.install_cached_token_capture()

        async def call_wrapped_create():
            tool_use_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return tool_use_quality_module.get_cached_tokens()

        assert asyncio.run(call_wrapped_create()) == 31
        tool_use_quality_module.install_cached_token_capture()

    def test_cached_token_capture_survives_tracing_recovery(self, monkeypatch):
        """Recovery of a tracing wrapper must not drop cached-token capture."""
        from openai.resources.chat.completions import AsyncCompletions

        async def untraced_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 53}}}

        async def traced_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 51}}}

        traced_create._original = untraced_create

        monkeypatch.setattr(AsyncCompletions, "create", traced_create)
        tool_use_quality_module.install_cached_token_capture()
        installed_create = AsyncCompletions.create

        async def call_create():
            tool_use_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return tool_use_quality_module.get_cached_tokens()

        assert asyncio.run(call_create()) == 51

        # Simulate another concurrent run recovering the pre-tracing method.
        AsyncCompletions.create = installed_create._original

        assert asyncio.run(call_create()) == 53

    def test_cached_token_capture_reuses_existing_wrapper(self, monkeypatch):
        """A second install registers this module's context var on the existing wrapper."""
        from openai.resources.chat.completions import AsyncCompletions

        async def raw_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 71}}}

        monkeypatch.setattr(AsyncCompletions, "create", raw_create)
        tool_use_quality_module.install_cached_token_capture()
        first_wrapper = AsyncCompletions.create

        tool_use_quality_module.install_cached_token_capture()

        assert AsyncCompletions.create is first_wrapper

        async def call_create():
            tool_use_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return tool_use_quality_module.get_cached_tokens()

        assert asyncio.run(call_create()) == 71

    def test_cached_token_capture_install_ignores_missing_create(self, monkeypatch):
        """A missing OpenAI create method is ignored during capture installation."""
        from openai.resources.chat.completions import AsyncCompletions

        monkeypatch.setattr(AsyncCompletions, "create", None)
        tool_use_quality_module.install_cached_token_capture()

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
        assert result["tool_use_quality_threshold"] == 1
        assert result["tool_use_quality_result"] == "fail"
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["score"] == 3
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["threshold"] == 4
        assert result["tool_use_quality_evaluators"]["tool_call_accuracy"]["passed"] is False
        assert result["tool_use_quality_evaluators"]["tool_selection"]["score"] == 1
        assert result["tool_call_accuracy_result"] == "fail"

    @pytest.mark.parametrize("score", [True, 6])
    def test_invalid_member_score_raises(self, score):
        """Boolean and out-of-range member scores raise evaluation exceptions."""
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
        assert result["tool_use_quality_threshold"] == 1
        assert result["tool_use_quality_result"] == "not_applicable"
        assert result["tool_use_quality_passed"] is None
        for name in _EVALUATOR_NAMES:
            assert result["tool_use_quality_evaluators"][name]["score"] is None
            assert result["tool_use_quality_evaluators"][name]["threshold"] == next(
                evaluator["default_threshold"] for evaluator in _EVALUATORS if evaluator["name"] == name
            )
            assert result["tool_use_quality_evaluators"][name]["status"] == "skipped"
            assert result["tool_use_quality_evaluators"][name]["passed"] is None
            assert result[f"{name}_score"] is None
            assert result[f"{name}_status"] == "skipped"
            assert result[f"{name}_result"] == "not_applicable"
            assert f"{name}_passed" not in result

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

    def test_missing_tool_definitions_are_skipped_per_evaluator_in_llm_result(self):
        """Missing definitions reach the LLM and only dependent members are skipped."""
        evaluator = _mock_flows(_make_evaluator(), _llm_output_without_tool_definitions())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["tool_use_quality_status"] == "completed"
        for name in ("tool_call_accuracy", "tool_input_accuracy", "tool_selection"):
            assert result[f"{name}_score"] is None
            assert result[f"{name}_status"] == "skipped"
            assert result[f"{name}_result"] == "not_applicable"
        for name in ("tool_call_success", "tool_output_utilization"):
            assert result[f"{name}_status"] == "completed"
        evaluator._flow.assert_called_once()
        assert evaluator._flow.call_args.kwargs["tool_definitions"] == []
        evaluator._multi_turn_flow.assert_not_called()

    def test_missing_tool_definitions_are_skipped_per_evaluator_for_messages(self):
        """Missing definitions reach the multi-turn LLM and permit member-level skips."""
        evaluator = _mock_flows(_make_evaluator(), _llm_output_without_tool_definitions())
        result = evaluator(messages=VALID_MESSAGES)
        assert result["tool_use_quality_status"] == "completed"
        for name in ("tool_call_accuracy", "tool_input_accuracy", "tool_selection"):
            assert result[f"{name}_score"] is None
            assert result[f"{name}_status"] == "skipped"
            assert result[f"{name}_result"] == "not_applicable"
        for name in ("tool_call_success", "tool_output_utilization"):
            assert result[f"{name}_status"] == "completed"
        evaluator._multi_turn_flow.assert_called_once()
        assert evaluator._multi_turn_flow.call_args.kwargs["tool_definitions"] == "TOOL_DEFINITIONS:"
        evaluator._flow.assert_not_called()

    def test_empty_tool_definitions_reach_llm_for_member_level_skips(self):
        """An empty definition list is sent to the LLM instead of short-circuiting."""
        evaluator = _mock_flows(_make_evaluator(), _llm_output_without_tool_definitions())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE, tool_definitions=[])
        assert result["tool_use_quality_status"] == "completed"
        for name in ("tool_call_accuracy", "tool_input_accuracy", "tool_selection"):
            assert result[f"{name}_status"] == "skipped"
        for name in ("tool_call_success", "tool_output_utilization"):
            assert result[f"{name}_status"] == "completed"
        evaluator._flow.assert_called_once()
        assert evaluator._flow.call_args.kwargs["tool_definitions"] == []

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
        """The direct evaluation path requires both query and response."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"response": VALID_RESPONSE, "tool_definitions": VALID_TOOL_DEFINITIONS}))

    def test_do_eval_missing_tool_definitions_reaches_llm(self):
        """The direct evaluation path sends missing definitions to member evaluation."""
        evaluator = _mock_flows(_make_evaluator(), _llm_output_without_tool_definitions())
        result = asyncio.run(evaluator._do_eval({"query": VALID_QUERY, "response": VALID_RESPONSE}))

        assert result["tool_use_quality_status"] == "completed"
        assert result["tool_call_accuracy_status"] == "skipped"
        assert result["tool_call_success_status"] == "completed"
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_super_real_call_handles_empty_multiple_and_conversion_errors(self):
        """The shared call path handles empty, multiple, and invalid converted inputs."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        td = VALID_TOOL_DEFINITIONS

        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[])
        empty_result = asyncio.run(evaluator._the_super_real_call(tool_definitions=td))
        assert empty_result["tool_use_quality"] is None
        assert empty_result["tool_use_quality_result"] == "not_applicable"

        evaluator._convert_kwargs_to_eval_input = MagicMock(return_value=[{"response": VALID_RESPONSE}] * 2)
        evaluator._do_eval = AsyncMock(return_value={"tool_use_quality": 1})
        evaluator._aggregate_results = MagicMock(return_value={"tool_use_quality": 1})
        assert asyncio.run(evaluator._the_super_real_call(tool_definitions=td)) == {"tool_use_quality": 1}

        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("invalid input"))
        with pytest.raises(ValueError, match="invalid input"):
            asyncio.run(evaluator._the_super_real_call(tool_definitions=td))

    # endregion


def test_aggregate_preserves_member_properties_and_shared_usage():
    """Preserve arbitrary member properties without multiplying shared model usage."""
    evaluator = _make_evaluator()
    members = _all_completed_llm_output()["llm_output"]
    properties = {"decision_model": {"native_answer": {"noul": 0.75}, "rubric_hash": "fixture"}}
    members["tool_selection"]["properties"] = properties
    result = evaluator._build_aggregate_result(members, {"total_tokens": 34})
    assert result["tool_use_quality_evaluators"]["tool_selection"]["properties"] == properties
    assert result["tool_use_quality_properties"]["total_tokens"] == 34
    assert result["tool_use_quality_total_tokens"] == 34
    assert "threshold" not in members["tool_selection"]


_native_decision_CONFIG = {"type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
                           "azure_deployment": "Decision-1", "api_key": "test-only"}
_native_decision_INPUTS = {"query": "Weather in Paris?", "response": "It is sunny.",
                           "tool_definitions": [{"name": "weather", "parameters": {}}]}
_native_decision_MESSAGES = [{"role": "user", "content": "Weather in Paris?"},
                             {"role": "assistant", "content": "It is sunny."}]
_native_decision_NAMES = [member["name"] for member in tool_use_quality_module._EVALUATORS]
_native_decision_PROMPTY_LOAD = "azure.ai.evaluation._legacy.prompty.AsyncPrompty.load"


def _native_decision_response():
    answers = {}
    for name in _native_decision_NAMES:
        answers[f"{name}_status"] = {"type": "noul", "noul": 1}
        answers[name] = {"type": "noul", "noul": 0.75}
    answers["tool_call_accuracy"] = {
        "type": "score", "score": 4, "confidence": 1,
        "probabilities": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1},
    }
    return {"model": "Decision-1", "answers": answers,
            "usage": {"input_tokens": 30, "output_tokens": 4}}


def _native_decision_evaluate(payload, inputs=None, **options):
    with patch(_native_decision_PROMPTY_LOAD, side_effect=AssertionError("Prompty must not initialize")) as load:
        evaluator = ToolUseQualityEvaluator(_native_decision_CONFIG, **options)
    load.assert_not_called()
    evaluator._decision_model_runner._post = AsyncMock(return_value=payload)
    result = evaluator(**(inputs if inputs is not None else _native_decision_INPUTS))
    evaluator._decision_model_runner._post.assert_awaited_once()
    return evaluator, result


def _native_decision_request(post):
    content, = post.call_args.args
    assert isinstance(content, bytes)
    body = json.loads(content)
    assert set(body) == {"model", "state", "questions"}
    assert body["model"] == _native_decision_CONFIG["azure_deployment"]
    return body["state"], body["questions"]


@pytest.mark.parametrize("level", [None, "turn", "conversation"])
@pytest.mark.parametrize("input_kind", ["query", "messages"])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_one_request_ten_questions_and_metadata(level, input_kind):
    """Route either input shape through one transport call and preserve provenance."""
    inputs = _native_decision_INPUTS if input_kind == "query" else {
        "messages": _native_decision_MESSAGES, "tool_definitions": _native_decision_INPUTS["tool_definitions"]
    }
    evaluator, result = _native_decision_evaluate(_native_decision_response(), inputs, evaluation_level=level)
    state, questions = _native_decision_request(evaluator._decision_model_runner._post)
    assert set(questions) == set(_native_decision_NAMES) | {f"{name}_status" for name in _native_decision_NAMES}
    assert len(questions) == 10
    assert sum(question["type"] == "score" for question in questions.values()) == 1
    conversation = level == "conversation" or (level is None and input_kind == "messages")
    assert ("messages" in state) == conversation
    strategies = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
    for name, strategy in strategies.items():
        assert questions[name] == strategy.questions[name]
        assert questions[f"{name}_status"] == strategy.questions["status"]
        member = result["tool_use_quality_evaluators"][name]
        provenance = member["properties"]["decision_model"]
        assert provenance["deployment"] == "Decision-1"
        assert provenance["native_answer"] == _native_decision_response()["answers"][name]
        assert provenance["status_completion_probability"] == 1
        assert len(provenance["rubric_hash"]) == 64
        assert len(provenance["status_rubric_hash"]) == 64
        assert "total_tokens" not in member["properties"]
        assert "failed_turn" not in member
        assert "No model rationale is available" in member["reason"]
    assert result["tool_use_quality_properties"]["total_tokens"] == 34
    assert result["tool_use_quality_total_tokens"] == 34
    assert result["tool_use_quality_properties"]["prompt_tokens"] == 30
    assert result["tool_use_quality_properties"]["completion_tokens"] == 4
    assert result["tool_use_quality"] == 1
    assert not hasattr(evaluator, "_flow")
    assert not hasattr(evaluator, "_multi_turn_flow")


@pytest.mark.parametrize("name", _native_decision_NAMES)
@pytest.mark.parametrize("status", [0, 0.4999])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_independent_skips_ignore_unusable_scores(name, status):
    """Skip only the inapplicable member before inspecting its malformed score."""
    payload = _native_decision_response()
    payload["answers"][f"{name}_status"]["noul"] = status
    payload["answers"][name] = {"type": "score" if name == "tool_call_accuracy" else "noul"}
    _, result = _native_decision_evaluate(payload)
    assert result[f"{name}_score"] is None
    assert result[f"{name}_status"] == "skipped"
    assert result[f"{name}_result"] == "not_applicable"
    assert result["tool_use_quality_properties"]["skipped_evaluators"] == [name]
    assert result["tool_use_quality"] == 1
    assert result["tool_use_quality_total_tokens"] == 34


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_all_skipped_keeps_request_usage():
    """Preserve usage even when no member is applicable."""
    payload = _native_decision_response()
    for name in _native_decision_NAMES:
        payload["answers"][f"{name}_status"]["noul"] = 0
        payload["answers"][name] = {"type": "score" if name == "tool_call_accuracy" else "noul"}
    _, result = _native_decision_evaluate(payload)
    assert result["tool_use_quality"] is None
    assert result["tool_use_quality_status"] == "skipped"
    assert result["tool_use_quality_result"] == "not_applicable"
    assert result["tool_use_quality_total_tokens"] == 34


@pytest.mark.parametrize("name", _native_decision_NAMES[1:])
@pytest.mark.parametrize("probability,score", [(0, 0), (0.7499, 0), (0.75, 1), (1, 1)])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_binary_boundary_and_any_fail(name, probability, score):
    """Use the runner cutoff independently and fail the aggregate on any member failure."""
    payload = _native_decision_response()
    payload["answers"][name]["noul"] = probability
    payload["answers"][f"{name}_status"]["noul"] = 0.5
    _, result = _native_decision_evaluate(payload)
    assert result[f"{name}_score"] == score
    assert result["tool_use_quality"] == score
    assert result["tool_use_quality_properties"]["failed_evaluators"] == ([] if score else [name])


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_likert_argmax_and_threshold(level):
    """Map zero-based native levels to the member's one-based scale, not the mean."""
    payload = _native_decision_response()
    payload["answers"]["tool_call_accuracy"]["probabilities"] = {
        str(index): int(index == level - 1) for index in range(5)
    }
    _, result = _native_decision_evaluate(payload)
    assert result["tool_call_accuracy_score"] == level
    assert result["tool_use_quality"] == int(level >= 3)


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_likert_ties_select_lower_level_and_override_threshold():
    """Respect deterministic ties and the composite's per-member threshold overrides."""
    payload = _native_decision_response()
    payload["answers"]["tool_call_accuracy"]["probabilities"] = {
        "0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0,
    }
    _, result = _native_decision_evaluate(payload, threshold={"tool_call_accuracy": 4})
    assert result["tool_call_accuracy_score"] == 3
    assert result["tool_call_accuracy_result"] == "fail"
    assert result["tool_use_quality"] == 0


@pytest.mark.parametrize("cutoff", [0, 0.5, 1])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_probability_override_does_not_mutate_config(cutoff):
    """Apply constructor cutoffs only to binary members without changing caller configuration."""
    evaluator, _ = _native_decision_evaluate(_native_decision_response(), probability_threshold=cutoff)
    assert "probability_threshold" not in _native_decision_CONFIG
    assert evaluator._decision_model_runner.probability_threshold == cutoff
    assert evaluator._binary_decision is True
    nested = {**_native_decision_CONFIG, "probability_threshold": 0.25}
    assert ToolUseQualityEvaluator(nested)._decision_model_runner.probability_threshold == 0.25


@pytest.mark.parametrize("cutoff", [-0.1, 1.1, True, "0.75", float("nan")])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_invalid_cutoff_rejected(cutoff):
    """Reject invalid binary cutoffs before execution."""
    with pytest.raises(EvaluationException, match="probability_threshold"):
        ToolUseQualityEvaluator(_native_decision_CONFIG, probability_threshold=cutoff)


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_chat_cutoff_and_older_sdk_rejected(monkeypatch):
    """Reject Decision-only settings and SDKs without shared parsing support."""
    with pytest.raises(EvaluationException, match="only supported for Decision"):
        ToolUseQualityEvaluator({"azure_endpoint": "https://example.openai.azure.com",
                                 "azure_deployment": "chat"}, probability_threshold=0.75)
    monkeypatch.setattr(tool_use_quality_module, "_SDK_SUPPORTS_DECISION", False)
    with patch(_native_decision_PROMPTY_LOAD, side_effect=AssertionError("Must reject before loading")):
        with pytest.raises(EvaluationException, match="parse support"):
            ToolUseQualityEvaluator(_native_decision_CONFIG)


@pytest.mark.parametrize("name", _native_decision_NAMES)
@pytest.mark.parametrize("invalid_status", [False, -1, 2, "bad", float("inf")])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_invalid_status_raises(name, invalid_status):
    """Validate every independent applicability probability."""
    payload = _native_decision_response()
    payload["answers"][f"{name}_status"]["noul"] = invalid_status
    with pytest.raises(EvaluationException, match="invalid response"):
        _native_decision_evaluate(payload)


@pytest.mark.parametrize("name", _native_decision_NAMES)
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_completed_unusable_score_raises(name):
    """Do not silently skip malformed scores for applicable members."""
    payload = _native_decision_response()
    payload["answers"][name] = {"type": "score" if name == "tool_call_accuracy" else "noul"}
    with pytest.raises(EvaluationException, match="invalid response"):
        _native_decision_evaluate(payload)


@pytest.mark.parametrize("defect", ["missing_answer", "wrong_type", "usage", "model"])
@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_malformed_transport_response_raises(defect):
    """Reject responses that fail the real runner's structural validation."""
    payload = _native_decision_response()
    if defect == "missing_answer":
        del payload["answers"]["tool_selection_status"]
    elif defect == "wrong_type":
        payload["answers"]["tool_selection"]["type"] = "score"
    elif defect == "usage":
        payload["usage"]["input_tokens"] = -1
    else:
        payload["model"] = ""
    with pytest.raises(EvaluationException, match="invalid response"):
        _native_decision_evaluate(payload)


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_transport_error_propagates_once():
    """Propagate a failed request without issuing member requests."""
    evaluator = ToolUseQualityEvaluator(_native_decision_CONFIG)
    evaluator._decision_model_runner._post = AsyncMock(side_effect=RuntimeError("transport failed"))
    with pytest.raises(RuntimeError, match="transport failed"):
        evaluator(**_native_decision_INPUTS)
    evaluator._decision_model_runner._post.assert_awaited_once()


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_async_and_missing_definitions():
    """Allow member-level applicability without definitions on the async entry point."""
    evaluator = ToolUseQualityEvaluator(_native_decision_CONFIG)
    evaluator._decision_model_runner._post = AsyncMock(return_value=_native_decision_response())
    result = asyncio.run(evaluator._to_async()(
        query=_native_decision_INPUTS["query"], response=_native_decision_INPUTS["response"]))
    evaluator._decision_model_runner._post.assert_awaited_once()
    state, _ = _native_decision_request(evaluator._decision_model_runner._post)
    assert state["tool_definitions"] == []
    assert result["tool_use_quality"] == 1


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_one_request_per_converted_input():
    """Batch each converted input separately rather than each metric separately."""
    evaluator = ToolUseQualityEvaluator(_native_decision_CONFIG)
    evaluator._decision_model_runner._post = AsyncMock(return_value=_native_decision_response())
    with patch.object(
        evaluator, "_convert_kwargs_to_eval_input",
        return_value=[dict(_native_decision_INPUTS), dict(_native_decision_INPUTS)],
    ):
        with patch.object(evaluator, "_aggregate_results", side_effect=lambda **kwargs: kwargs["per_turn_results"]):
            results = asyncio.run(evaluator._the_super_real_call(**_native_decision_INPUTS))
    assert evaluator._decision_model_runner._post.await_count == 2
    assert len(results) == 2
    for call in evaluator._decision_model_runner._post.await_args_list:
        assert len(json.loads(call.args[0])["questions"]) == 10
    assert [result["tool_use_quality_total_tokens"] for result in results] == [34, 34]


@pytest.mark.skipif(
    not tool_use_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_rubrics_and_init_schema():
    """Validate both local rubrics and public named-deployment configuration."""
    root = Path(tool_use_quality_module.__file__).parent
    for filename in (ToolUseQualityEvaluator._DECISION_FILE, ToolUseQualityEvaluator._MULTI_TURN_DECISION_FILE):
        rubric = yaml.safe_load((root / filename).read_text(encoding="utf-8"))
        assert set(rubric) == set(_native_decision_NAMES)
        for name, member in rubric.items():
            assert set(member) == {"status", "result"}
            assert member["status"]["type"] == "noul"
            assert set(member["status"]["criteria"]) == {"true", "false"}
            assert member["result"]["type"] == ("score" if name == "tool_call_accuracy" else "noul")
            assert len(member["result"]["criteria"]) == (5 if name == "tool_call_accuracy" else 2)
    spec = yaml.safe_load((root.parent / "spec.yaml").read_text(encoding="utf-8"))
    assert spec["version"] == 8
    schema = spec["initParameterSchema"]
    config = {key: value for key, value in _native_decision_CONFIG.items() if key != "api_key"}
    assert "model_config" not in schema["properties"]
    for valid in ({"deployment_name": "chat"},
                  {"deployment_name": "Decision-1", "probability_threshold": 0.75}):
        jsonschema.validate(valid, schema)
    assert "not" not in schema
    for invalid in ({},
                    {"model_config": {"type": "decision"}},
                    {"model_config": {**config, "probability_threshold": 2}},
                    {"model_config": config, "probability_threshold": -1},
                    {"deployment_name": "Decision-1", "probability_threshold": -1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
