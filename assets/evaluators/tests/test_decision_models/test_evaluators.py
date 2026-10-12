"""Verify opt-in evaluator calls and unchanged validation without live models."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import ToolCallAccuracyEvaluator
from .test_backend import CONFIG

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
    evaluator = TaskCompletionEvaluator(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Must not call"))
    result = evaluator(query="Hello", response=[{"role": "assistant", "content": [{"type": "function_call",
                       "name": "weather", "tool_call_id": "call-1", "arguments": {}}]}],
                       tool_definitions=[{"name": "weather", "parameters": {}}])
    assert result["task_completion_status"] == "skipped"
    post.assert_not_called()


def test_accuracy_score_and_metadata():
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
    assert state["tool_calls"] == calls


def test_accuracy_model_status_skips_before_parsing_score():
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
    evaluator = ToolCallAccuracyEvaluator(CONFIG)
    post = stub(evaluator, side_effect=AssertionError("Must not call"))
    result = evaluator(query="Hello", tool_definitions=[{"name": "weather", "parameters": {}}],
                       response=[{"role": "assistant", "content": [{"type": "text", "text": "Hello"}]}])
    assert result["tool_call_accuracy_status"] == "skipped"
    post.assert_not_called()


@pytest.mark.parametrize("evaluator_class", [TaskCompletionEvaluator, ToolCallAccuracyEvaluator])
def test_bad_answer_raises_not_scored_fail(evaluator_class):
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
    legacy = {"backend": "decision", "provider": "jev", "model": "m", "endpoint": "https://x.example", "api_key": "k"}
    with pytest.raises(EvaluationException):
        evaluator_class(legacy)
