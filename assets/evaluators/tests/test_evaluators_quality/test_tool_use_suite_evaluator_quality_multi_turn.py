# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for the Tool Use Evaluation Suite (multi-turn) with real flow execution.

See test_tool_use_suite_evaluator_quality.py for why this file builds the evaluator
directly instead of using BaseQualityEvaluatorRunner.

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


FLIGHT_TOOL_DEFINITIONS = [
    {
        "name": "search_flights",
        "description": "Search for flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["origin", "destination"],
        },
    },
    {
        "name": "book_flight",
        "description": "Book a specific flight by id.",
        "parameters": {
            "type": "object",
            "properties": {"flight_id": {"type": "string"}},
            "required": ["flight_id"],
        },
    },
]


@pytest.mark.quality
class TestToolUseEvaluationSuiteQualityMultiTurn:
    """Quality tests for ToolUseEvaluationSuite with multi-turn (messages) input."""

    def test_all_evaluators_pass_across_multi_turn_session(self):
        """A correctly executed, multi-turn flight-booking session passes all five evaluators."""
        evaluator = _make_real_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Find me a flight from NYC to London."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "search_flights",
                        "arguments": {"origin": "NYC", "destination": "London"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "tool_result", "tool_result": {"flight_id": "BA123", "price": 450}}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I found a British Airways flight (BA123) for $450. Book it?"}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Yes, please book it."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "book_flight",
                        "arguments": {"flight_id": "BA123"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [{"type": "tool_result", "tool_result": {"status": "confirmed"}}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Your flight BA123 is booked and confirmed!"}],
            },
        ]
        result = evaluator(messages=messages, tool_definitions=FLIGHT_TOOL_DEFINITIONS)
        for name in (
            "tool_call_accuracy",
            "tool_call_success",
            "tool_input_accuracy",
            "tool_output_utilization",
            "tool_selection",
        ):
            _assert_evaluator_passed(result, name)
            assert result["tool_use_suite_evaluators"][name]["failed_turn"] is None
        assert result["tool_use_suite"] == 1

    def test_tool_output_utilization_fails_and_localizes_failed_turn(self):
        """A later turn that misuses an earlier tool output fails tool_output_utilization with a failed_turn."""
        evaluator = _make_real_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Find me a flight from NYC to London."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "search_flights",
                        "arguments": {"origin": "NYC", "destination": "London"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [
                    {"type": "tool_result", "tool_result": {"flight_id": "BA123", "price": 450}}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I found a British Airways flight (BA123) for $450. Book it?"}
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "Yes, please book it."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_2",
                        "name": "book_flight",
                        "arguments": {"flight_id": "BA123"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "content": [{"type": "tool_result", "tool_result": {"status": "confirmed"}}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Your flight BA123 is booked and confirmed for $900!"}
                ],
            },
        ]
        result = evaluator(messages=messages, tool_definitions=FLIGHT_TOOL_DEFINITIONS)
        _assert_evaluator_failed(result, "tool_output_utilization")
        assert result["tool_use_suite_evaluators"]["tool_output_utilization"]["failed_turn"] == 1
        assert result["tool_use_suite"] == 0
