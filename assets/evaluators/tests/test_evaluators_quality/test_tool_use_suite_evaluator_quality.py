# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for the Tool Use Evaluation Suite (single-turn) with real flow execution.

ToolUseEvaluationSuite batches five evaluators into a single LLM call, so its output shape
(one primary score plus five raw evaluator result objects) differs from the single-evaluator
shape assumed by ``BaseQualityEvaluatorRunner``. This file therefore builds the evaluator
directly (same env-var + ``DefaultAzureCredential`` pattern used by
``BasePromptyEvaluatorRunner._init_evaluator``) and asserts against each evaluator's own
result key.

Requires real Azure OpenAI credentials via environment variables:
``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_DEPLOYMENT``, ``AZURE_OPENAI_API_VERSION``,
plus ``DefaultAzureCredential``-compatible auth (e.g. ``az login``).
"""

import os

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.identity import DefaultAzureCredential

from ...builtin.tool_use_suite.evaluator._tool_use_suite import ToolUseEvaluationSuite


def _make_real_evaluator() -> ToolUseEvaluationSuite:
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
    )
    return ToolUseEvaluationSuite(model_config=model_config, credential=DefaultAzureCredential())


def _assert_evaluator_passed(result, name: str):
    evaluator = result["tool_use_suite_evaluators"][name]
    assert evaluator["status"] == "completed", f"{name} should be completed, got {evaluator.get('status')}"
    assert evaluator["score"] is not None, f"{name} expected a score, reason: {evaluator.get('reason')}"


def _assert_evaluator_failed(result, name: str):
    evaluator = result["tool_use_suite_evaluators"][name]
    assert evaluator["status"] == "completed", f"{name} should be completed, got {evaluator.get('status')}"
    assert evaluator["score"] is not None, f"{name} expected a score, reason: {evaluator.get('reason')}"


def _assert_evaluator_skipped(result, name: str):
    evaluator = result["tool_use_suite_evaluators"][name]
    assert evaluator["status"] == "skipped", f"{name} expected skipped, got {evaluator.get('status')}"
    assert evaluator["score"] is None


WEATHER_TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "Fetches current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

WEATHER_QUERY = [
    {"role": "user", "content": [{"type": "text", "text": "What's the weather in Seattle?"}]}
]


@pytest.mark.quality
class TestToolUseEvaluationSuiteQuality:
    """Quality tests for ToolUseEvaluationSuite with single-turn (query/response) input."""

    def test_all_evaluators_pass_for_correct_single_tool_call(self):
        """A single correct, successful, well-utilized, well-selected tool call passes all five evaluators."""
        evaluator = _make_real_evaluator()
        result = evaluator(
            query=WEATHER_QUERY,
            response=[
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
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The weather in Seattle is rainy at 14 degrees C."}],
                },
            ],
            tool_definitions=WEATHER_TOOL_DEFINITIONS,
        )
        for name in (
            "tool_call_accuracy",
            "tool_call_success",
            "tool_input_accuracy",
            "tool_output_utilization",
            "tool_selection",
        ):
            _assert_evaluator_passed(result, name)
        assert result["tool_use_suite"] == 1
        assert result["tool_use_suite_evaluators"]["tool_call_accuracy"]["score"] == 5

    def test_tool_input_accuracy_fails_on_fabricated_parameter(self):
        """A tool call with a fabricated parameter (not present in the conversation) fails tool_input_accuracy."""
        evaluator = _make_real_evaluator()
        result = evaluator(
            query=WEATHER_QUERY,
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": {"weather": "Sunny, 22C"}}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The weather in Paris is sunny at 22 degrees C."}],
                },
            ],
            tool_definitions=WEATHER_TOOL_DEFINITIONS,
        )
        _assert_evaluator_failed(result, "tool_input_accuracy")
        assert result["tool_use_suite"] == 0

    def test_tool_call_success_fails_on_tool_error(self):
        """A tool call whose result indicates a technical error fails tool_call_success."""
        evaluator = _make_real_evaluator()
        result = evaluator(
            query=WEATHER_QUERY,
            response=[
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
                    "content": [{"type": "tool_result", "tool_result": "Error: upstream weather service timed out"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Sorry, I couldn't retrieve the weather right now."}],
                },
            ],
            tool_definitions=WEATHER_TOOL_DEFINITIONS,
        )
        _assert_evaluator_failed(result, "tool_call_success")
        assert result["tool_use_suite"] == 0

    def test_tool_selection_fails_when_wrong_tool_used(self):
        """Selecting a tool irrelevant to the user's request fails tool_selection."""
        evaluator = _make_real_evaluator()
        tool_definitions = WEATHER_TOOL_DEFINITIONS + [
            {
                "name": "send_email",
                "description": "Sends an email to a recipient.",
                "parameters": {
                    "type": "object",
                    "properties": {"to": {"type": "string"}, "body": {"type": "string"}},
                    "required": ["to", "body"],
                },
            }
        ]
        result = evaluator(
            query=WEATHER_QUERY,
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "send_email",
                            "arguments": {"to": "user@example.com", "body": "Checking weather for you."},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": "Email sent."}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I've emailed you about the weather."}],
                },
            ],
            tool_definitions=tool_definitions,
        )
        _assert_evaluator_failed(result, "tool_selection")
        assert result["tool_use_suite"] == 0

    def test_all_evaluators_skipped_when_no_tool_calls_made(self):
        """No tool calls in the response yields skipped/not_applicable results for tool-dependent evaluators."""
        evaluator = _make_real_evaluator()
        result = evaluator(
            query=[{"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}],
            response=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'm doing well, thank you! How can I help you today?"}],
                }
            ],
            tool_definitions=WEATHER_TOOL_DEFINITIONS,
        )
        for name in (
            "tool_call_accuracy",
            "tool_call_success",
            "tool_input_accuracy",
            "tool_selection",
        ):
            _assert_evaluator_skipped(result, name)
