# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Verify opt-in evaluator calls and unchanged validation without live models."""

import asyncio
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallAccuracyEvaluator
from ..test_evaluators_behavior import common_tool_test_data as data

CONFIG = {
    "type": "decision",
    "azure_endpoint": "https://example.services.ai.azure.com",
    "azure_deployment": "Decision-1",
    "api_key": "test-only",
}
UNUSABLE = {
    "task_completion": {"type": "noul", "noul": "unusable"},
    "tool_call_accuracy": {"type": "score", "probabilities": "unusable"},
}
PROMPTY_LOAD = "azure.ai.evaluation._evaluators._common._model_runners.AsyncPrompty.load"


def native(question_id, answer, completion_probability=1):
    """Build a native provider result with an applicability decision.

    A skipped item carries a structurally valid but unusable answer, which must never be read.
    """
    answers = {question_id: UNUSABLE[question_id] if answer is None else answer}
    answers["status"] = {"type": "noul", "noul": completion_probability}
    return {"model": "actual-model", "answers": answers,
            "usage": {"input_tokens": 30, "output_tokens": 4}}


def stub(evaluator, payload=None, **kwargs):
    """Replace the evaluator's single transport call and return the stub for assertions."""
    evaluator._model_runner._post = AsyncMock(return_value=payload, **kwargs)
    return evaluator._model_runner._post


def test_completion_sync_without_prompty():
    """Evaluate task completion synchronously without loading a chat flow."""
    with patch(PROMPTY_LOAD, side_effect=AssertionError("Chat flow must not be initialized")):
        evaluator = TaskCompletionEvaluator(CONFIG)
    post = stub(evaluator, native("task_completion", {"type": "noul", "noul": 0.9}))
    result = evaluator(query="What is the capital of France?", response="Paris.")
    assert result["task_completion"] == 1
    assert result["task_completion_threshold"] == 1
    assert result["task_completion_status"] == "completed"
    assert result["task_completion_properties"]["model"] == "actual-model"
    assert result["task_completion_properties"]["total_tokens"] == 34
    assert "task_requirements" not in result["task_completion_properties"]
    assert result["task_completion_properties"]["decision_model"]["deployment"] == "Decision-1"
    assert "No model rationale" in result["task_completion_reason"]
    state, questions = post.call_args.args
    assert set(state) == {"query", "response", "tool_definitions"}
    assert questions["task_completion"]["type"] == "noul"
    assert questions["status"]["type"] == "noul"


@pytest.mark.parametrize("conversation", [False, True])
def test_completion_model_status_skips_before_parsing_score(conversation):
    """Skip inapplicable tasks before reading an unusable completion answer."""
    evaluator = TaskCompletionEvaluator(CONFIG)
    post = stub(evaluator, native("task_completion", None, 0.1))
    inputs = {"messages": [{"role": "user", "content": "Say hello"},
                           {"role": "assistant", "content": "Hello"}]} if conversation else {
        "query": "Say hello", "response": "Hello",
    }
    result = evaluator(**inputs)
    post.assert_awaited_once()
    assert result["task_completion_status"] == "skipped"
    assert result["task_completion_result"] == "not_applicable"
    assert result["task_completion"] is None
    assert result["task_completion_passed"] is None


def test_completion_conversation_and_cutoff():
    """Apply the configured probability cutoff to asynchronous conversation evaluation."""
    evaluator = TaskCompletionEvaluator({**CONFIG, "probability_threshold": 0.8})
    post = stub(evaluator, native("task_completion", {"type": "noul", "noul": 0.7}, 0.5))
    messages = [{"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "Hello"}]
    result = asyncio.run(evaluator._to_async()(messages=messages))
    assert result["task_completion"] == 0
    assert result["task_completion_result"] == "fail"
    state, questions = post.call_args.args
    assert "messages" in state
    assert "end to end" in questions["task_completion"]["instructions"]


def test_completion_intermediate_skips_without_call():
    """Skip intermediate tool responses without invoking the Decision model."""
    evaluator = TaskCompletionEvaluator(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Must not call"))
    result = evaluator(query="Hello", response=[{"role": "assistant", "content": [{"type": "function_call",
                       "name": "weather", "tool_call_id": "call-1", "arguments": {}}]}],
                       tool_definitions=[{"name": "weather", "parameters": {}}])
    assert result["task_completion_status"] == "skipped"
    post.assert_not_called()


def test_accuracy_score_and_metadata():
    """Map tied accuracy probabilities to the lower level and preserve model metadata."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG, threshold=4)
    answer = {"type": "score", "score": 2.5, "confidence": 0.8,
              "probabilities": {"0": 0, "1": 0, "2": 0.5, "3": 0.5, "4": 0}}
    post = stub(evaluator, native("tool_call_accuracy", answer))
    calls = [{"type": "tool_call", "name": "weather", "tool_call_id": "call-1", "arguments": {"city": "Paris"}}]
    result = evaluator(query="Weather in Paris?", tool_calls=calls,
                       tool_definitions=[{"name": "weather", "parameters": {}}])
    assert result["tool_call_accuracy"] == 3
    assert result["tool_call_accuracy_result"] == "fail"
    assert result["tool_call_accuracy_threshold"] == 4
    properties = result["tool_call_accuracy_properties"]
    assert properties["total_tokens"] == 34
    assert "tool_calls_made_by_agent" not in properties
    assert "correct_tool_calls_made_by_agent" not in properties
    assert properties["decision_model"]["deployment"] == "Decision-1"
    assert "tool_call_accuracy_total_tokens" not in result
    state, questions = post.call_args.args
    assert set(questions) == {"tool_call_accuracy", "status"}
    assert questions["tool_call_accuracy"]["type"] == "score"
    assert questions["status"]["type"] == "noul"
    assert set(state) == {"query", "tool_calls", "tool_definitions"}
    assert state["tool_calls"] == calls


def test_accuracy_model_status_skips_before_parsing_score():
    """Skip inapplicable tool calls before reading an unusable accuracy answer."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, native("tool_call_accuracy", None, 0.1))
    result = evaluator(
        query="Weather in Paris?",
        tool_definitions='[{"name": "weather", "parameters": {}}]',
        tool_calls=[{"type": "tool_call", "name": "lookup_weather", "tool_call_id": "call-1", "arguments": {}}],
    )
    post.assert_awaited_once()
    assert result["tool_call_accuracy_status"] == "skipped"
    assert result["tool_call_accuracy_result"] == "not_applicable"
    assert result["tool_call_accuracy"] is None
    assert result["tool_call_accuracy_passed"] is None
    assert result["tool_call_accuracy_properties"] is None


def test_accuracy_without_calls_skips():
    """Skip responses without tool calls without invoking the Decision model."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Must not call"))
    result = evaluator(query="Hello", tool_definitions=[{"name": "weather", "parameters": {}}],
                       response=[{"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}])
    assert result["tool_call_accuracy_status"] == "skipped"
    post.assert_not_called()


@pytest.mark.parametrize("selected", [False, True])
def test_accuracy_response_evidence_preserves_selection_and_retry(selected):
    """Keep result/retry evidence without adding context calls to an explicit selection."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, native("tool_call_accuracy", {
        "type": "score", "score": 3, "confidence": 1,
        "probabilities": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 0},
    }))
    calls = [
        {"type": "tool_call", "name": "weather", "tool_call_id": call_id, "arguments": {"city": "Paris"}}
        for call_id in ("failed", "retry")
    ]
    context_call = {"type": "tool_call", "name": "lookup_city", "tool_call_id": "context", "arguments": {}}
    response = [
        {"role": "assistant", "content": [deepcopy(context_call)]},
        {"role": "tool", "tool_call_id": "context",
         "content": [{"type": "tool_result", "tool_result": {"city": "Paris"}}]},
        {"role": "assistant", "content": [{**calls[0], "type": "function_call"}]},
        {"role": "tool", "tool_call_id": "failed",
         "content": [{"type": "function_call_output", "function_call_output": {"error": "timeout"}}]},
        {"role": "assistant", "content": [{**calls[1], "type": "function_call"}]},
        {"role": "tool", "tool_call_id": "retry",
         "content": [{"type": "function_call_output", "function_call_output": {"temperature": 20}}]},
        {"role": "assistant", "content": [{"type": "text", "text": "20 degrees"}]},
    ]
    definitions = [{"name": "weather", "parameters": {}}]
    inputs = {"tool_calls": deepcopy(calls)} if selected else {}
    if not selected:
        definitions.append({"name": "lookup_city", "parameters": {}})
    result = evaluator(query="Weather in Paris?", response=deepcopy(response),
                       tool_definitions=definitions, **inputs)
    assert result["tool_call_accuracy"] == 4
    state, questions = post.call_args.args
    assert set(state) == {"query", "tool_calls", "tool_definitions", "response"}
    assert state["query"] == "Weather in Paris?"
    assert state["tool_definitions"] == definitions
    if selected:
        assert state["tool_calls"] == calls
    else:
        assert [call["tool_call_id"] for call in state["tool_calls"]] == ["context", "failed", "retry"]
        assert state["tool_calls"][1]["tool_result"] == {"error": "timeout"}
        assert state["tool_calls"][2]["tool_result"] == {"temperature": 20}
    expected_response = deepcopy(response)
    for message in expected_response:
        for item in message["content"]:
            if item["type"] == "function_call":
                item["type"] = "tool_call"
            elif item["type"] == "function_call_output":
                item["type"] = "tool_result"
                item["tool_result"] = item.pop("function_call_output")
    assert state["response"] == expected_response
    assert "must not be scored" in questions["tool_call_accuracy"]["instructions"]


@pytest.mark.parametrize("tool", ["OPENAPI", "FILE_SEARCH"])
def test_accuracy_response_only_definitions_and_normalized_state(tool):
    """Preserve OpenAPI and built-in call extraction in native Decision state."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, native("tool_call_accuracy", {
        "type": "score", "score": 4, "confidence": 1,
        "probabilities": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1},
    }))
    query = deepcopy(getattr(data, f"{tool}_QUERY"))
    response = deepcopy(getattr(data, f"{tool}_RESPONSE"))
    definitions = deepcopy(getattr(data, f"{tool}_TOOL_DEFINITIONS"))
    evaluator(query=query, response=response, tool_definitions=definitions)
    state, _ = post.call_args.args
    expected_calls = data.FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS if tool == "FILE_SEARCH" else [{
        "type": "tool_call",
        "tool_call_id": "call_cfb55a91a58c44ea9217b34174aad2ab",
        "name": "weather_GetCurrentWeather",
        "arguments": {"location": "Cairo", "format": "j1"},
        "tool_result": "",
    }]
    assert state["tool_calls"] == expected_calls
    assert all(call["type"] == "tool_call" for call in state["tool_calls"])
    assert state["response"] == response
    assert {call["name"] for call in state["tool_calls"]} <= {
        definition["name"] for definition in state["tool_definitions"]
    }


def test_accuracy_intermediate_skips_without_call():
    """Retain the intermediate-response guard for Decision tool evaluation."""
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Must not call"))
    result = evaluator(
        query="Weather in Paris?",
        response=[{"role": "assistant", "content": [{
            "type": "function_call", "name": "weather", "tool_call_id": "call-1", "arguments": {},
        }]}],
        tool_definitions=[{"name": "weather", "parameters": {}}],
    )
    assert result["tool_call_accuracy_status"] == "skipped"
    post.assert_not_called()


@pytest.mark.parametrize("evaluation_level", [None, "turn", "conversation"])
@pytest.mark.parametrize("status_probability,score_probability,score", [
    (0.4999, 1, None), (0.5, 0.7499, 0), (0.5, 0.75, 1),
])
def test_completion_scope_and_default_cutoff_boundaries(
    evaluation_level, status_probability, score_probability, score,
):
    """Preserve applicability .5, completion .75, and public threshold 1 at each scope."""
    evaluator = TaskCompletionEvaluator(CONFIG, evaluation_level=evaluation_level)
    post = stub(evaluator, native(
        "task_completion", {"type": "noul", "noul": score_probability}, status_probability,
    ))
    messages = [
        {"role": "user", "content": "First task"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Latest task"},
        {"role": "assistant", "content": "Latest answer"},
    ]
    result = evaluator(messages=messages)
    assert result["task_completion"] == score
    assert result["task_completion_threshold"] == 1
    assert result["task_completion_status"] == ("skipped" if score is None else "completed")
    state, questions = post.call_args.args
    if evaluation_level == "turn":
        assert set(state) == {"query", "response", "tool_definitions"}
        assert "Latest task" in state["query"]
        assert "Latest answer" in state["response"]
        assert "First answer" not in state["response"]
        assert "end to end" not in questions["task_completion"]["instructions"]
    else:
        assert set(state) == {"messages"}
        assert all(text in state["messages"] for text in ("First task", "First answer", "Latest task", "Latest answer"))
        assert "end to end" in questions["task_completion"]["instructions"]


@pytest.mark.parametrize("probability,expected", [(0.7999, 0), (0.8, 1)])
def test_completion_configured_cutoff_is_inclusive(probability, expected):
    """The configured probability cutoff changes binary scoring, not the public threshold."""
    evaluator = TaskCompletionEvaluator({**CONFIG, "probability_threshold": 0.8})
    stub(evaluator, native("task_completion", {"type": "noul", "noul": probability}, 0.5))
    result = evaluator(query="Say hello", response="Hello")
    assert result["task_completion"] == expected
    assert result["task_completion_threshold"] == 1
    assert result["task_completion_passed"] is bool(expected)


def test_completion_forced_conversation_uses_full_query_response_transcript():
    """Forcing conversation scope preserves all history rather than selecting the latest turn."""
    evaluator = TaskCompletionEvaluator(CONFIG, evaluation_level="conversation")
    post = stub(evaluator, native("task_completion", {"type": "noul", "noul": 0.75}, 0.5))
    evaluator(
        query=[{"role": "user", "content": "First task"}, {"role": "assistant", "content": "First answer"},
               {"role": "user", "content": "Latest task"}],
        response=[{"role": "assistant", "content": "Latest answer"}],
    )
    state, questions = post.call_args.args
    assert set(state) == {"messages"}
    assert all(text in state["messages"] for text in ("First task", "First answer", "Latest task", "Latest answer"))
    assert "end to end" in questions["task_completion"]["instructions"]


@pytest.mark.parametrize("evaluator_class", [TaskCompletionEvaluator, ToolCallAccuracyEvaluator])
def test_bad_answer_raises_not_scored_fail(evaluator_class):
    """Raise an evaluation error for malformed answers instead of returning a failing score."""
    evaluator = evaluator_class(CONFIG)
    stub(evaluator, {})
    inputs = {"query": "Hello", "response": "Hello"} if evaluator_class is TaskCompletionEvaluator else {
        "query": "Weather in Paris?", "tool_definitions": [{"name": "weather", "parameters": {}}],
        "tool_calls": [{"type": "tool_call", "name": "weather", "tool_call_id": "call-1", "arguments": {}}],
    }
    with pytest.raises(EvaluationException, match="invalid response"):
        evaluator(**inputs)


@pytest.mark.parametrize("evaluator_class", [TaskCompletionEvaluator, ToolCallAccuracyEvaluator])
def test_legacy_backend_configuration_is_rejected(evaluator_class):
    """Reject legacy backend configuration fields for both evaluators."""
    legacy = {"backend": "decision", "provider": "jev", "model": "m", "endpoint": "https://x.example", "api_key": "k"}
    with pytest.raises(EvaluationException):
        evaluator_class(legacy)
