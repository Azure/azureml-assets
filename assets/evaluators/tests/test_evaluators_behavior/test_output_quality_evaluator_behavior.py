# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for the Output Quality Evaluator."""

import asyncio
import builtins
import importlib.util
from pathlib import Path
from copy import deepcopy
import os
import runpy
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jsonschema
import pytest
import yaml
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._evaluators import _common
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.output_quality.evaluator import _output_quality as output_quality_module
from ...builtin.output_quality.evaluator._output_quality import OutputQualityEvaluator, _EVALUATORS

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
    return OutputQualityEvaluator(model_config=model_config, **init_kwargs)


def _mock_flows(evaluator, llm_output):
    """Mock both prompty flows with the same LLM payload."""

    async def side_effect(timeout, **kwargs):
        return llm_output

    evaluator._flow = MagicMock(side_effect=side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=side_effect)
    return evaluator


@pytest.fixture
def decision_mocks(monkeypatch):
    """Mock Decision capabilities while initializing the released SDK's chat base."""
    base = output_quality_module.ModelEvaluatorBase
    original_init = base.__init__
    runner = AsyncMock(probability_threshold=0.8)
    configurations = []
    strategies = []

    def initialize_chat(instance, **kwargs):
        configurations.append(kwargs["model_config"])
        runner.probability_threshold = kwargs["model_config"].get("probability_threshold", 0.8)
        kwargs["model_config"] = {
            "azure_endpoint": "https://Sanitized.api.cognitive.microsoft.com",
            "azure_deployment": "aoai-deployment",
        }
        original_init(instance, **kwargs)

    def create_strategy(*args, **kwargs):
        name = kwargs["result_key"]
        member = next(member for member in _EVALUATORS if member["name"] == name)
        parsed = {
            f"{name}_score": member["max"],
            f"{name}_status": "completed",
            f"{name}_reason": f"{name} Decision reason",
            f"{name}_properties": {"decision_model": {"member": name}},
            f"{name}_result": "pass",
            f"{name}_passed": True,
            f"{name}_threshold": kwargs["threshold"],
            "unrelated_field": "must not leak",
        }
        strategy = Mock(questions=kwargs["questions"], parse=Mock(return_value=parsed))
        strategies.append(strategy)
        return strategy

    answers = {}
    for member in _EVALUATORS:
        name = member["name"]
        answers[f"{name}_status"] = {"type": "noul", "noul": 1}
        answers[name] = (
            {"type": "noul", "noul": 0.9} if member["max"] == 1 else
            {"type": "score", "score": 4, "confidence": 1,
             "probabilities": {str(index): float(index == 4) for index in range(5)}}
        )
    runner.return_value = {
        "answers": answers, "model": "decision-model",
        "usage": {"input_tokens": 30, "output_tokens": 4},
    }
    binary = Mock()
    likert = Mock()
    factory = Mock(side_effect=create_strategy)
    chat_load = Mock(return_value=AsyncMock())
    monkeypatch.setattr(base, "__init__", initialize_chat)
    monkeypatch.setattr(base, "_decision_model_runner", property(lambda instance: runner), raising=False)
    monkeypatch.setattr(output_quality_module, "_SDK_SUPPORTS_DECISION", True)
    monkeypatch.setattr(output_quality_module, "BinaryScoring", binary)
    monkeypatch.setattr(output_quality_module, "LikertScoring", likert)
    monkeypatch.setattr(output_quality_module, "DecisionEvaluationStrategy", factory)
    monkeypatch.setattr(output_quality_module.AsyncPrompty, "load", chat_load)
    return Mock(runner=runner, configurations=configurations, strategies=strategies,
                binary=binary, likert=likert, factory=factory, chat_load=chat_load)


def _make_decision_evaluator(**kwargs):
    """Construct a Decision evaluator without importing unreleased SDK helpers."""
    return OutputQualityEvaluator(
        model_config={"type": "decision", "azure_endpoint": "https://example.services.ai.azure.com",
                      "azure_deployment": "Decision-1"},
        **kwargs,
    )


@pytest.mark.unittest
class TestOutputQualityDecisionBehavior:
    """Exercise the composite's Decision code with local SDK capability mocks."""

    def test_constructor_loads_both_rubrics_without_multi_chat(self, decision_mocks):
        """Set up member strategies and return before loading multi-turn chat."""
        evaluator = _make_decision_evaluator(threshold={"fluency": 4})
        mocks = decision_mocks
        assert mocks.factory.call_count == 12
        assert mocks.binary.call_count == 4
        assert mocks.likert.call_count == 8
        for call in mocks.binary.call_args_list:
            assert call.args == (0.8,)
        for call in mocks.likert.call_args_list:
            assert call.args == (5,)
            assert call.kwargs == {"minimum": 1}
        for call in mocks.factory.call_args_list:
            name = call.kwargs["result_key"]
            assert call.args == (mocks.runner,)
            assert set(call.kwargs["questions"]) == {"status", name}
            assert call.kwargs["questions"]["status"]["type"] == "noul"
            assert call.kwargs["questions"][name]["type"] == (
                "noul" if name in {"task_adherence", "task_completion"} else "score"
            )
            assert call.kwargs["scoring"] is (
                mocks.binary.return_value if name in {"task_adherence", "task_completion"}
                else mocks.likert.return_value
            )
            assert call.kwargs["threshold"] == evaluator._threshold_for(name)
            assert call.kwargs["timeout"] == evaluator._LLM_CALL_TIMEOUT
            assert call.kwargs["error_target"] == (
                output_quality_module.ExtendedErrorTarget.OUTPUT_QUALITY_EVALUATOR
            )
        assert list(evaluator._decision_strategies) == _EVALUATOR_NAMES
        assert list(evaluator._multi_turn_decision_strategies) == _EVALUATOR_NAMES
        assert evaluator._decision_strategies["task_completion"].questions != (
            evaluator._multi_turn_decision_strategies["task_completion"].questions
        )
        mocks.runner.assert_not_awaited()
        mocks.chat_load.assert_called_once()
        assert not hasattr(evaluator, "_multi_turn_flow")

    @pytest.mark.parametrize("inputs,level,conversation", [
        ({"query": VALID_QUERY, "response": VALID_RESPONSE}, None, False),
        ({"messages": VALID_MESSAGES}, None, True),
        ({"query": VALID_QUERY, "response": VALID_RESPONSE}, "conversation", True),
        ({"messages": VALID_MESSAGES}, "turn", False),
        ({"response": VALID_RESPONSE}, None, False),
    ])
    def test_one_transport_maps_all_member_fields(self, decision_mocks, inputs, level, conversation):
        """Batch twelve questions and map isolated parser outputs through either route."""
        evaluator = _make_decision_evaluator(evaluation_level=level)
        result = evaluator(**deepcopy(inputs), tool_definitions=deepcopy(VALID_TOOL_DEFINITIONS))
        runner = decision_mocks.runner
        runner.assert_awaited_once()
        prepared = runner.await_args.kwargs["state"]
        questions = runner.await_args.kwargs["questions"]
        assert runner.await_args.kwargs["timeout"] == evaluator._LLM_CALL_TIMEOUT
        assert len(questions) == 12
        assert set(questions) == set(_EVALUATOR_NAMES) | {f"{name}_status" for name in _EVALUATOR_NAMES}
        assert ("messages" in prepared) is conversation
        if conversation:
            assert "User turn 1:" in prepared["messages"]
            assert "Agent turn 1:" in prepared["messages"]
            assert prepared["tool_definitions"]
        elif "messages" not in inputs:
            assert prepared["query"] == inputs.get("query", [])
            assert prepared["response"] == VALID_RESPONSE
        selected = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
        unselected = evaluator._decision_strategies if conversation else evaluator._multi_turn_decision_strategies
        for name, strategy in selected.items():
            assert questions[f"{name}_status"] == strategy.questions["status"]
            assert questions[name] == strategy.questions[name]
            strategy.parse.assert_called_once_with({
                **runner.return_value,
                "answers": {"status": runner.return_value["answers"][f"{name}_status"],
                            name: runner.return_value["answers"][name]},
            }, prepared)
            unselected[name].parse.assert_not_called()
            member = result["output_quality_evaluators"][name]
            for field in ("score", "status", "reason", "properties", "result", "passed", "threshold"):
                assert member[field] == strategy.parse.return_value[f"{name}_{field}"]
            assert "unrelated_field" not in member
            for field in ("score", "reason", "threshold", "status", "result"):
                assert result[f"{name}_{field}"] == member[field]
            assert f"{name}_passed" not in result
        assert result["output_quality"] == result["output_quality_score"] == 1
        assert result["output_quality_result"] == "pass"
        assert result["output_quality_passed"] is True
        assert result["output_quality_status"] == "completed"
        assert result["output_quality_threshold"] == 1
        assert result["output_quality_reason"] == "All quality evaluators passed."
        for field, expected in {"prompt_tokens": 30, "completion_tokens": 4,
                                "total_tokens": 34, "model": "decision-model"}.items():
            assert result[f"output_quality_{field}"] == expected
            assert result["output_quality_properties"][field] == expected
        evaluator._flow.assert_not_called()
        assert not hasattr(evaluator, "_multi_turn_flow")

    @pytest.mark.parametrize("outcome", ["fail", "mixed", "skipped"])
    @pytest.mark.parametrize("conversation", [False, True])
    def test_decision_aggregation(self, decision_mocks, outcome, conversation):
        """Aggregate parsed failures and skips while retaining one request's usage."""
        evaluator = _make_decision_evaluator()
        selected = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
        affected = _EVALUATOR_NAMES if outcome == "skipped" else ["fluency"]
        for name in affected:
            parsed = selected[name].parse.return_value
            parsed[f"{name}_score"] = 2 if outcome == "fail" else None
            parsed[f"{name}_status"] = "completed" if outcome == "fail" else "skipped"
            parsed.pop(f"{name}_passed")
            parsed.pop(f"{name}_result")
            parsed.pop(f"{name}_threshold")
        inputs = {"messages": deepcopy(VALID_MESSAGES)} if conversation else {
            "query": VALID_QUERY, "response": VALID_RESPONSE,
        }
        result = evaluator(**inputs)
        decision_mocks.runner.assert_awaited_once()
        assert result["output_quality"] == {"fail": 0, "mixed": 1, "skipped": None}[outcome]
        assert result["output_quality_result"] == {
            "fail": "fail", "mixed": "pass", "skipped": "not_applicable",
        }[outcome]
        assert result["output_quality_properties"]["failed_evaluators"] == (
            ["fluency"] if outcome == "fail" else []
        )
        assert result["output_quality_properties"]["skipped_evaluators"] == (
            [] if outcome == "fail" else affected
        )
        assert result["output_quality_total_tokens"] == 34
        member = result["output_quality_evaluators"]["fluency"]
        assert member["passed"] is (False if outcome == "fail" else None)
        assert member["threshold"] == 3
        if outcome != "fail":
            assert "fluency_result" not in result

    @pytest.mark.parametrize("override", [None, 0.7, 0.0, 1.0])
    def test_cutoff_forwarding_preserves_input_config(self, decision_mocks, override):
        """Forward nested or explicit cutoff without changing the caller's configuration."""
        config = {"type": "decision", "probability_threshold": 0.9}
        original = dict(config)
        evaluator = OutputQualityEvaluator(
            config, probability_threshold=override, threshold={"fluency": 4},
        )
        cutoff = 0.9 if override is None else override
        assert config == original
        assert decision_mocks.configurations == [{**config, "probability_threshold": cutoff}]
        for call in decision_mocks.binary.call_args_list:
            assert call.args == (cutoff,)
        assert evaluator._threshold_for("fluency") == 4
        assert evaluator._threshold_for("task_completion") == 1

    @pytest.mark.parametrize("cutoff", [None, 0.8])
    def test_unsupported_capability_fails_before_base_init(self, monkeypatch, cutoff):
        """Reject Decision configuration on a released SDK without parse support."""
        monkeypatch.setattr(output_quality_module, "_SDK_SUPPORTS_DECISION", False)
        base_init = Mock(side_effect=AssertionError("Base must not initialize"))
        monkeypatch.setattr(output_quality_module.ModelEvaluatorBase, "__init__", base_init)
        with pytest.raises(EvaluationException, match="decision parse support"):
            _make_decision_evaluator(probability_threshold=cutoff)
        base_init.assert_not_called()

    def test_released_sdk_chat_without_new_capabilities(self, monkeypatch):
        """Load the evaluator with no new SDK symbols and keep chat scoring functional."""
        monkeypatch.delattr(_common, "ModelEvaluatorBase", raising=False)
        monkeypatch.setitem(sys.modules, "azure.ai.evaluation._evaluators._common._decision_strategy", None)
        namespace = runpy.run_path(output_quality_module.__file__)
        assert namespace["ModelEvaluatorBase"] is _common.PromptyEvaluatorBase
        assert namespace["BinaryScoring"] is namespace["LikertScoring"] is None
        assert namespace["DecisionEvaluationStrategy"] is None
        assert not namespace["_SDK_SUPPORTS_DECISION"]
        evaluator = namespace["OutputQualityEvaluator"]({
            "azure_endpoint": "https://Sanitized.api.cognitive.microsoft.com",
            "azure_deployment": "aoai-deployment",
        })
        _mock_flows(evaluator, _all_completed_llm_output())
        assert evaluator(query=VALID_QUERY, response=VALID_RESPONSE)["output_quality"] == 1
        with pytest.raises(EvaluationException, match="decision parse support"):
            namespace["OutputQualityEvaluator"]({"type": "decision"})

    def test_broken_sdk_dependency_is_not_hidden(self, monkeypatch):
        """Propagate unrelated SDK import errors instead of hiding a broken install."""
        original_import = builtins.__import__
        error = ModuleNotFoundError(name="broken_dependency")

        def broken_import(name, *args, **kwargs):
            if name == "azure.ai.evaluation._evaluators._common._decision_strategy":
                raise error
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", broken_import)
        with pytest.raises(ModuleNotFoundError) as caught:
            runpy.run_path(output_quality_module.__file__)
        assert caught.value is error


@pytest.mark.unittest
class TestOutputQualityEvaluatorBehavior:
    """Behavioral tests for the OutputQualityEvaluator."""

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

    def test_primary_score_and_member_results_are_flattened(self):
        """Aggregate output includes nested member results and converter-compatible flat keys."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["output_quality"] == 1
        assert result["output_quality_threshold"] == 1
        assert result["output_quality_result"] == "pass"
        assert result["output_quality_passed"] is True
        evaluators = result["output_quality_evaluators"]
        assert "evaluators" not in result["output_quality_properties"]
        assert evaluators["fluency"]["threshold"] == 3
        assert evaluators["fluency"]["passed"] is True
        for name in _EVALUATOR_NAMES:
            assert name not in result
            assert evaluators[name]["status"] == "completed"
            for field in ("score", "reason", "threshold", "status"):
                assert result[f"{name}_{field}"] == evaluators[name][field]
            assert result[f"{name}_result"] == "pass"
            assert f"{name}_passed" not in result

    @pytest.mark.parametrize("input_kwargs", [
        {"query": VALID_QUERY, "response": VALID_RESPONSE},
        {"messages": VALID_MESSAGES},
    ])
    def test_cached_tokens_are_exposed_for_both_flow_types(self, input_kwargs):
        """Cached prompt tokens are available to the SDK converter for either Prompty flow."""
        llm_output = _all_completed_llm_output()
        llm_output["cached_tokens"] = 128
        evaluator = _mock_flows(_make_evaluator(), llm_output)

        result = evaluator(**input_kwargs)

        assert result["output_quality_cached_tokens"] == 128
        assert result["output_quality_properties"]["cached_tokens"] == 128
        assert (
            result["output_quality_properties"][
                output_quality_module.PROMPT_CACHE_CACHED_TOKENS_PROPERTY
            ]
            == 128
        )

    def test_cached_token_capture_helpers_are_self_contained(self, monkeypatch):
        """The evaluator file captures cached tokens without package-local imports."""

        class Details:
            cached_tokens = 19

        class Usage:
            prompt_tokens_details = Details()

        class Response:
            usage = Usage()

        assert output_quality_module._extract_cached_tokens(
            {"usage": {"prompt_tokens_details": {"cached_tokens": 17}}}
        ) == 17
        assert output_quality_module._extract_cached_tokens(Response()) == 19
        assert output_quality_module._extract_cached_tokens({}) is None
        assert output_quality_module._extract_cached_tokens({"usage": {}}) is None
        assert output_quality_module._extract_cached_tokens(
            {"usage": {"prompt_tokens_details": {"cached_tokens": True}}}
        ) is None

        from openai.resources.chat.completions import AsyncCompletions

        async def fake_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 23}}}

        monkeypatch.setattr(AsyncCompletions, "create", fake_create)
        output_quality_module.install_cached_token_capture()

        async def call_wrapped_create():
            output_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return output_quality_module.get_cached_tokens()

        assert asyncio.run(call_wrapped_create()) == 23
        output_quality_module.install_cached_token_capture()

    def test_cached_token_capture_survives_tracing_recovery(self, monkeypatch):
        """Recovery of a tracing wrapper must not drop cached-token capture."""
        from openai.resources.chat.completions import AsyncCompletions

        async def untraced_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 43}}}

        async def traced_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 41}}}

        traced_create._original = untraced_create

        monkeypatch.setattr(AsyncCompletions, "create", traced_create)
        output_quality_module.install_cached_token_capture()
        installed_create = AsyncCompletions.create

        async def call_create():
            output_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return output_quality_module.get_cached_tokens()

        assert asyncio.run(call_create()) == 41

        # Simulate another concurrent run recovering the pre-tracing method.
        AsyncCompletions.create = installed_create._original

        assert asyncio.run(call_create()) == 43

    def test_cached_token_capture_reuses_existing_wrapper(self, monkeypatch):
        """A second install registers this module's context var on the existing wrapper."""
        from openai.resources.chat.completions import AsyncCompletions

        async def raw_create(self):
            return {"usage": {"prompt_tokens_details": {"cached_tokens": 61}}}

        monkeypatch.setattr(AsyncCompletions, "create", raw_create)
        output_quality_module.install_cached_token_capture()
        first_wrapper = AsyncCompletions.create

        output_quality_module.install_cached_token_capture()

        assert AsyncCompletions.create is first_wrapper

        async def call_create():
            output_quality_module.clear_cached_tokens()
            await AsyncCompletions.create(object())
            return output_quality_module.get_cached_tokens()

        assert asyncio.run(call_create()) == 61

    def test_cached_token_capture_install_ignores_missing_create(self, monkeypatch):
        """A missing OpenAI create method is ignored during capture installation."""
        from openai.resources.chat.completions import AsyncCompletions

        monkeypatch.setattr(AsyncCompletions, "create", None)
        output_quality_module.install_cached_token_capture()

    def test_multi_turn_raw_failed_turn_is_preserved(self):
        """Multi-turn raw evaluator results preserve their failed-turn metadata."""
        evaluator = _mock_flows(_make_evaluator(), _all_completed_llm_output(failed_turn=1))
        result = evaluator(messages=VALID_MESSAGES)
        for name in _EVALUATOR_NAMES:
            assert result["output_quality_evaluators"][name]["failed_turn"] == 1

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

    def test_chat_rejects_decision_probability_threshold(self):
        """Reject Decision-only cutoff options on an unchanged chat configuration."""
        with pytest.raises(EvaluationException, match="only supported for Decision"):
            _make_evaluator(probability_threshold=0.8)

    def test_raw_member_properties_and_skipped_unusable_scores_are_preserved(self):
        """Preserve raw provenance while ignoring unusable scores on skipped members."""
        llm_output = _all_completed_llm_output()
        provenance = {"decision_model": {"native_answer": {"type": "noul", "noul": 0.9}}}
        llm_output["llm_output"]["task_completion"]["properties"] = provenance
        llm_output["llm_output"]["groundedness"] = {
            "score": "unusable", "status": "skipped", "reason": "Not applicable.",
        }
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["output_quality"] == 1
        assert result["output_quality_evaluators"]["task_completion"]["properties"] == provenance
        assert result["output_quality_evaluators"]["groundedness"]["score"] == "unusable"
        assert result["output_quality_evaluators"]["groundedness"]["passed"] is None

    def test_any_member_below_threshold_fails_primary_score(self):
        """A member below its threshold fails the aggregate result."""
        evaluator = _mock_flows(
            _make_evaluator(threshold={"fluency": 4}),
            _all_completed_llm_output(score_overrides={"fluency": 3}),
        )
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["output_quality"] == 0
        assert result["output_quality_threshold"] == 1
        assert result["output_quality_result"] == "fail"
        assert result["output_quality_passed"] is False
        assert result["output_quality_evaluators"]["fluency"]["score"] == 3
        assert result["output_quality_evaluators"]["fluency"]["threshold"] == 4
        assert result["output_quality_evaluators"]["fluency"]["passed"] is False
        assert result["output_quality_evaluators"]["coherence"]["score"] == 5
        assert result["fluency_result"] == "fail"

    # endregion

    # region skipped/not-applicable handling

    def test_all_members_skipped_returns_not_applicable(self):
        """An all-skipped member result returns an aggregate not-applicable result."""
        evaluator = _mock_flows(_make_evaluator(), _all_skipped_llm_output())
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["output_quality"] is None
        assert result["output_quality_threshold"] == 1
        assert result["output_quality_result"] == "not_applicable"
        assert result["output_quality_status"] == "skipped"
        assert result["output_quality_passed"] is None
        for name in _EVALUATOR_NAMES:
            assert result["output_quality_evaluators"][name]["score"] is None
            assert result["output_quality_evaluators"][name]["threshold"] == next(
                evaluator["default_threshold"] for evaluator in _EVALUATORS if evaluator["name"] == name
            )
            assert result["output_quality_evaluators"][name]["status"] == "skipped"
            assert result["output_quality_evaluators"][name]["passed"] is None
            assert result[f"{name}_score"] is None
            assert result[f"{name}_status"] == "skipped"
            assert f"{name}_result" not in result
            assert f"{name}_passed" not in result

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
        assert result["output_quality"] is None
        assert result["output_quality_status"] == "skipped"
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
        assert result["output_quality"] == 1
        assert result["output_quality_evaluators"]["groundedness"]["status"] == "skipped"
        assert result["output_quality_evaluators"]["fluency"]["status"] == "completed"

    def test_missing_member_output_is_treated_as_skipped(self):
        """An omitted member output is represented as a skipped evaluator."""
        llm_output = _all_completed_llm_output()
        del llm_output["llm_output"]["task_completion"]
        evaluator = _mock_flows(_make_evaluator(), llm_output)
        result = evaluator(query=VALID_QUERY, response=VALID_RESPONSE)
        assert result["output_quality_evaluators"]["task_completion"]["score"] is None
        assert result["output_quality_evaluators"]["task_completion"]["status"] == "skipped"
        assert result["output_quality_evaluators"]["fluency"]["status"] == "completed"

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
        evaluator._do_eval = AsyncMock(return_value={"output_quality": 1})
        evaluator._aggregate_results = MagicMock(return_value={"output_quality": 1})
        assert asyncio.run(evaluator._the_super_real_call()) == {"output_quality": 1}

        evaluator._convert_kwargs_to_eval_input = MagicMock(side_effect=ValueError("invalid input"))
        with pytest.raises(ValueError, match="invalid input"):
            asyncio.run(evaluator._the_super_real_call())

    # endregion


_native_decision_CONFIG = {
    "type": "decision",
    "azure_endpoint": "https://example.services.ai.azure.com",
    "azure_deployment": "Decision-1",
    "api_key": "test-only",
}
_native_decision_INPUTS = {"query": "Say hello", "response": "Hello"}
_native_decision_MESSAGES = [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello"}]
_native_decision_NAMES = [member["name"] for member in _EVALUATORS]
_native_decision_BINARY = {"task_adherence", "task_completion"}
_native_decision_PROMPTY_LOAD = "azure.ai.evaluation._evaluators._common._model_runners.AsyncPrompty.load"


def _native_decision_score(level=5):
    return {"type": "score", "score": level - 1, "confidence": 1,
            "probabilities": {str(index): float(index == level - 1) for index in range(5)}}


def _native_decision_native(skipped=(), overrides=None):
    answers = {}
    for name in _native_decision_NAMES:
        answers[f"{name}_status"] = {"type": "noul", "noul": 0.1 if name in skipped else 1}
        answers[name] = {"type": "noul", "noul": 0.9} if name in _native_decision_BINARY else _native_decision_score()
        if name in skipped:
            answers[name] = {"type": answers[name]["type"], "noul": "unusable", "probabilities": "unusable"}
    answers.update(overrides or {})
    return {"model": "actual-model", "answers": answers, "usage": {"input_tokens": 30, "output_tokens": 4}}


def _native_decision_stub(evaluator, payload=None, **kwargs):
    evaluator._decision_model_runner._post = AsyncMock(return_value=payload, **kwargs)
    return evaluator._decision_model_runner._post


@pytest.mark.parametrize("execution", ["sync", "async", "sync_in_loop"])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_one_transport_twelve_questions_and_no_chat(execution):
    """Run all members with one transport and preserve native metadata in raw members."""
    with patch(_native_decision_PROMPTY_LOAD, side_effect=AssertionError("No chat initialization")), patch.object(
        output_quality_module.AsyncPrompty, "load", side_effect=AssertionError("No multi-turn chat initialization")
    ):
        evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, _native_decision_native())
    if execution == "sync":
        result = evaluator(**_native_decision_INPUTS)
    else:
        async def invoke():
            if execution == "async":
                return await evaluator._to_async()(**_native_decision_INPUTS)
            return evaluator(**_native_decision_INPUTS)
        result = asyncio.run(invoke())
    post.assert_awaited_once()
    state, questions = post.call_args.args
    assert state["query"] == _native_decision_INPUTS["query"]
    assert state["response"] == _native_decision_INPUTS["response"]
    assert set(questions) == set(_native_decision_NAMES) | {f"{name}_status" for name in _native_decision_NAMES}
    assert len(questions) == 12
    assert not hasattr(evaluator, "_flow") and not hasattr(evaluator, "_multi_turn_flow")
    assert result["output_quality"] == 1
    assert result["output_quality_total_tokens"] == 34
    assert result["output_quality_properties"]["total_tokens"] == 34
    for name in _native_decision_NAMES:
        member = result["output_quality_evaluators"][name]
        provenance = member["properties"]["decision_model"]
        assert provenance["deployment"] == "Decision-1"
        assert provenance["model"] == "actual-model"
        assert provenance["native_answer"] == _native_decision_native()["answers"][name]
        assert provenance["native_status_answer"] == _native_decision_native()["answers"][f"{name}_status"]
        assert len(provenance["rubric_hash"]) == 64 and len(provenance["status_rubric_hash"]) == 64
        assert "No model rationale" in member["reason"]
        assert "failed_turn" not in member
        assert result[f"{name}_score"] == member["score"]
        assert f"{name}_total_tokens" not in result


@pytest.mark.parametrize("name", _native_decision_NAMES)
@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_independent_skip_ignores_unusable_score(name, conversation):
    """Gate every member separately before interpreting an unusable score."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, _native_decision_native(skipped=[name]))
    result = evaluator(messages=deepcopy(_native_decision_MESSAGES)
                       ) if conversation else evaluator(**_native_decision_INPUTS)
    post.assert_awaited_once()
    assert result["output_quality"] == 1
    assert result["output_quality_properties"]["skipped_evaluators"] == [name]
    member = result["output_quality_evaluators"][name]
    assert member["score"] is None and member["passed"] is None and member["status"] == "skipped"
    assert f"{name}_result" not in result
    assert all(result[f"{other}_status"] == "completed" for other in _native_decision_NAMES if other != name)
    state, questions = post.call_args.args
    strategies = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
    assert questions[f"{name}_status"] == strategies[name].questions["status"]
    assert ("messages" in state) is conversation


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_all_skipped_still_counts_single_request_tokens():
    """Return not applicable when every member skips without losing actual usage."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, _native_decision_native(skipped=_native_decision_NAMES))
    result = evaluator(**_native_decision_INPUTS)
    post.assert_awaited_once()
    assert result["output_quality"] is None
    assert result["output_quality_passed"] is None
    assert result["output_quality_status"] == "skipped"
    assert result["output_quality_result"] == "not_applicable"
    assert result["output_quality_total_tokens"] == 34


@pytest.mark.parametrize("name", _native_decision_NAMES)
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_any_completed_member_failure_fails_aggregate(name):
    """Fail the composite when any evaluated member is below its public threshold."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    answer = {"type": "noul", "noul": 0.74} if name in _native_decision_BINARY else _native_decision_score(2)
    _native_decision_stub(evaluator, _native_decision_native(overrides={name: answer}))
    result = evaluator(**_native_decision_INPUTS)
    assert result["output_quality"] == 0
    assert result["output_quality_properties"]["failed_evaluators"] == [name]
    assert result[f"{name}_result"] == "fail"


@pytest.mark.parametrize("probability,expected", [(0.7499, 0), (0.75, 1)])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_default_binary_cutoff(probability, expected):
    """Use the default inclusive probability cutoff for both binary members."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    _native_decision_stub(evaluator, _native_decision_native(
        overrides={name: {"type": "noul", "noul": probability} for name in _native_decision_BINARY}))
    result = evaluator(**_native_decision_INPUTS)
    for name in _native_decision_BINARY:
        assert result[f"{name}_score"] == expected
        assert result["output_quality_evaluators"][name]["properties"]["decision_model"][
            "probability_threshold"] == 0.75


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_cutoff_override_does_not_mutate_configuration_or_public_thresholds():
    """Override nested binary cutoff independently of Likert and public score thresholds."""
    config = {**_native_decision_CONFIG, "probability_threshold": 0.9}
    evaluator = OutputQualityEvaluator(config, probability_threshold=0.7, threshold={"fluency": 4})
    _native_decision_stub(evaluator, _native_decision_native(
        overrides={"task_completion": {"type": "noul", "noul": 0.7}}))
    result = evaluator(**_native_decision_INPUTS)
    assert config["probability_threshold"] == 0.9
    assert result["task_completion_score"] == 1
    assert result["task_completion_threshold"] == 1
    assert result["fluency_threshold"] == 4
    assert result["output_quality_threshold"] == 1


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_nested_cutoff_and_likert_tie():
    """Use nested binary cutoff and select the lower one-based Likert level on ties."""
    evaluator = OutputQualityEvaluator(
        {**_native_decision_CONFIG, "probability_threshold": 0.95}, threshold={"fluency": 4})
    answer = {"type": "score", "score": 2.5, "confidence": 0.8,
              "probabilities": {"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}}
    _native_decision_stub(evaluator, _native_decision_native(
        overrides={"fluency": answer, "fluency_status": {"type": "noul", "noul": 0.5}}))
    result = evaluator(**_native_decision_INPUTS)
    assert result["fluency_score"] == 3
    assert result["fluency_result"] == "fail"
    assert result["task_completion_score"] == 0
    provenance = result["output_quality_evaluators"]["fluency"]["properties"]["decision_model"]
    assert provenance["native_mean"] == 2.5 and provenance["expected_score"] == 3.5


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_likert_levels(level):
    """Map all native score levels to the existing one-to-five public scale."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    _native_decision_stub(evaluator, _native_decision_native(overrides={name: _native_decision_score(
        level) for name in _native_decision_NAMES if name not in _native_decision_BINARY}))
    result = evaluator(**_native_decision_INPUTS)
    for name in set(_native_decision_NAMES) - _native_decision_BINARY:
        assert result[f"{name}_score"] == level


@pytest.mark.parametrize("inputs,level,conversation", [
    (_native_decision_INPUTS, None, False), ({"messages": _native_decision_MESSAGES}, None, True),
    (_native_decision_INPUTS, "conversation", True), ({"messages": _native_decision_MESSAGES}, "turn", False),
])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_routing_selects_own_rubrics(inputs, level, conversation):
    """Select turn or conversation rubrics after existing input preparation."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG, evaluation_level=level)
    post = _native_decision_stub(evaluator, _native_decision_native())
    evaluator(**deepcopy(inputs))
    state, questions = post.call_args.args
    assert ("messages" in state) is conversation
    strategies = evaluator._multi_turn_decision_strategies if conversation else evaluator._decision_strategies
    for name in _native_decision_NAMES:
        assert questions[name] == strategies[name].questions[name]
    if conversation:
        assert "User turn 1:" in state["messages"] and "Agent turn 1:" in state["messages"]


@pytest.mark.parametrize("conversation", [False, True])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_response_only_retains_all_applicability_questions(conversation):
    """Keep fluency applicable without silently removing other members' status questions."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, _native_decision_native(skipped=set(_native_decision_NAMES) - {"fluency"}))
    inputs = {"messages": [{"role": "assistant", "content": "Hello"}]} if conversation else {"response": "Hello"}
    result = asyncio.run(evaluator._do_eval(inputs)) if conversation else evaluator(**inputs)
    state, questions = post.call_args.args
    assert len(questions) == 12
    applicability = questions["fluency_status"]["criteria"]["true"]
    assert ("No user turn is required" if conversation else "NOT required") in applicability
    assert result["output_quality"] == 1 and result["fluency_score"] == 5
    assert len(result["output_quality_properties"]["skipped_evaluators"]) == 5
    if not conversation:
        assert state["query"] == []


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, "0.5", float("nan")])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_invalid_probability_configuration(bad):
    """Reject invalid cutoffs through the shared binary runner configuration check."""
    with pytest.raises(EvaluationException, match="probability_threshold"):
        OutputQualityEvaluator(_native_decision_CONFIG, probability_threshold=bad)


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_unsupported_sdk_rejects_decision_before_loading_chat(monkeypatch):
    """Reject Decision clearly when the required no-transport parse capability is unavailable."""
    monkeypatch.setattr(output_quality_module, "_SDK_SUPPORTS_DECISION", False)
    with patch(_native_decision_PROMPTY_LOAD, side_effect=AssertionError("No chat initialization")):
        with pytest.raises(EvaluationException, match="decision parse support"):
            OutputQualityEvaluator(_native_decision_CONFIG)


@pytest.mark.parametrize("change", ["missing_answer", "bad_status", "bad_binary", "bad_likert", "bad_usage"])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_invalid_response_raises(change):
    """Reject malformed structures or applicable answer values without additional transport."""
    payload = _native_decision_native()
    if change == "missing_answer":
        del payload["answers"]["fluency"]
    elif change == "bad_status":
        payload["answers"]["fluency_status"]["noul"] = "bad"
    elif change == "bad_binary":
        payload["answers"]["task_completion"]["noul"] = 2
    elif change == "bad_likert":
        payload["answers"]["fluency"]["probabilities"] = "bad"
    else:
        payload["usage"]["input_tokens"] = -1
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, payload)
    with pytest.raises(EvaluationException):
        evaluator(**_native_decision_INPUTS)
    post.assert_awaited_once()


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_timeout_propagates_without_additional_request():
    """Surface the runner timeout without retrying through member strategies."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, side_effect=asyncio.TimeoutError)
    with pytest.raises(EvaluationException, match="timed out"):
        evaluator(**_native_decision_INPUTS)
    post.assert_awaited_once()


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_intermediate_response_skips_without_transport():
    """Preserve the existing early skip for intermediate tool responses."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, side_effect=AssertionError("No call"))
    result = evaluator(query="Hello", response=[{
        "role": "assistant", "content": [{
            "type": "function_call", "name": "lookup", "tool_call_id": "call-1", "arguments": {},
        }],
    }])
    assert result["output_quality_status"] == "skipped"
    post.assert_not_called()


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_decision_usage_cannot_capture_stale_chat_cache():
    """Clear chat cache telemetry before accounting for a Decision request."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    _native_decision_stub(evaluator, _native_decision_native())
    output_quality_module._CACHED_TOKENS.set(1000)
    result = asyncio.run(evaluator._do_eval(dict(_native_decision_INPUTS)))
    assert "cached_tokens" not in result["output_quality_properties"]
    assert result["output_quality_total_tokens"] == 34


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_spec_initializer_is_exclusive_and_validates_cutoffs():
    """Require exactly one initializer and validate Decision deployment and probability fields."""
    path = Path(output_quality_module.__file__).parents[1] / "spec.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    schema = spec["initParameterSchema"]
    jsonschema.validate({"deployment_name": "chat"}, schema)
    config = {key: value for key, value in _native_decision_CONFIG.items() if key != "api_key"}
    jsonschema.validate({"model_config": config, "probability_threshold": 0.75}, schema)
    for invalid in ({}, {"deployment_name": "chat", "model_config": config},
                    {"model_config": {"type": "decision"}},
                    {"model_config": {**config, "azure_endpoint": ""}},
                    {"model_config": {**config, "probability_threshold": 2}},
                    {"model_config": config, "probability_threshold": -1}):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_rubrics_keep_distinct_applicability_and_scales():
    """Keep all six own rubrics, including response-only and multi-turn skip distinctions."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    single = evaluator._decision_strategies
    multi = evaluator._multi_turn_decision_strategies
    for strategies in (single, multi):
        assert set(strategies) == set(_native_decision_NAMES)
        for name, strategy in strategies.items():
            assert set(strategy.questions) == {"status", name}
            assert strategy.questions["status"]["type"] == "noul"
            result = strategy.questions[name]
            assert result["type"] == ("noul" if name in _native_decision_BINARY else "score")
            assert len(result["criteria"]) == (2 if name in _native_decision_BINARY else 5)
    assert "unrelated topic jumps" in multi["coherence"].questions["status"]["criteria"]["false"]
    assert "no substantive factual claims" in multi["groundedness"].questions["status"]["criteria"]["false"]
    assert "including responses with no verifiable factual claims" in single["groundedness"].questions[
        "status"]["criteria"]["true"]


@pytest.mark.parametrize("missing", ["module", "parse", "base"])
@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_guarded_sdk_capabilities(missing, monkeypatch):
    """Keep chat available with missing Decision capabilities and reject Decision early."""
    import azure.ai.evaluation._evaluators._common as sdk_common

    original_import = builtins.__import__
    strategy_module = "azure.ai.evaluation._evaluators._common._decision_strategy"

    def guarded_import(name, *args, **kwargs):
        if name == strategy_module and missing == "module":
            raise ModuleNotFoundError(name=strategy_module)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    if missing == "parse":
        monkeypatch.setattr(output_quality_module.DecisionEvaluationStrategy, "parse", None)
    elif missing == "base":
        monkeypatch.delattr(sdk_common, "ModelEvaluatorBase")
    spec = importlib.util.spec_from_file_location("output_quality_capability_probe", output_quality_module.__file__)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe._SDK_SUPPORTS_DECISION is False
    expected_base = sdk_common.PromptyEvaluatorBase if missing == "base" else sdk_common.ModelEvaluatorBase
    assert issubclass(probe.OutputQualityEvaluator, expected_base)
    with patch(_native_decision_PROMPTY_LOAD, return_value=AsyncMock()), patch.object(probe.AsyncPrompty, "load"):
        chat = probe.OutputQualityEvaluator({
            "azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "chat",
        })
    chat._flow = AsyncMock(return_value={"llm_output": {
        member["name"]: {"score": member["max"], "status": "completed", "reason": "Fixture"}
        for member in _EVALUATORS
    }})
    assert chat(**_native_decision_INPUTS)["output_quality"] == 1
    with pytest.raises(EvaluationException, match="decision parse support"):
        probe.OutputQualityEvaluator(_native_decision_CONFIG)


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_guarded_import_does_not_hide_dependency_failure(monkeypatch):
    """Reraise internal SDK dependency failures rather than claiming Decision is unavailable."""
    original_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "azure.ai.evaluation._evaluators._common._decision_strategy":
            raise ModuleNotFoundError(name="missing_internal_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    spec = importlib.util.spec_from_file_location("output_quality_dependency_probe", output_quality_module.__file__)
    probe = importlib.util.module_from_spec(spec)
    with pytest.raises(ModuleNotFoundError) as error:
        spec.loader.exec_module(probe)
    assert error.value.name == "missing_internal_dependency"


@pytest.mark.skipif(
    not output_quality_module._SDK_SUPPORTS_DECISION, reason="Shared Decision SDK support is unavailable."
)
def test_native_decision_one_runner_request_per_converted_input():
    """Batch the six members per input without batching away separate input evaluations."""
    evaluator = OutputQualityEvaluator(_native_decision_CONFIG)
    post = _native_decision_stub(evaluator, _native_decision_native())
    with (
        patch.object(evaluator, "_convert_kwargs_to_eval_input", return_value=[
                     dict(_native_decision_INPUTS), dict(_native_decision_INPUTS)]),
        patch.object(evaluator, "_aggregate_results", return_value={"output_quality": 1}) as aggregate,
    ):
        result = asyncio.run(evaluator._the_super_real_call())
    assert result == {"output_quality": 1}
    assert post.await_count == 2
    assert all(len(call.args[1]) == 12 for call in post.await_args_list)
    results = aggregate.call_args.kwargs["per_turn_results"]
    assert all(result["output_quality_total_tokens"] == 34 for result in results)
