# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Efficiency Evaluator."""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.task_efficiency.evaluator._task_efficiency import (
    TaskEfficiencyEvaluator,
    serialize_messages,
    _count_assistant_tool_calls,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


def _create_mocked_evaluator(**init_kwargs):
    """Create a TaskEfficiencyEvaluator with its _flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = TaskEfficiencyEvaluator(model_config=model_config, **init_kwargs)
    evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator("task_efficiency"))
    return evaluator


# A trajectory with a single assistant tool call - enough to exercise the LLM path.
VALID_MESSAGES_WITH_TOOLS: List[Dict[str, Any]] = [
    {"role": "user", "content": [{"type": "text", "text": "Add a --verbose flag to the CLI."}]},
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "name": "read_file",
                "arguments": {"path": "src/cli.py"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [{"type": "tool_result", "tool_result": "parser = argparse.ArgumentParser()"}],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "Added the --verbose flag."}]},
]

VALID_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
        },
    }
]


@pytest.mark.unittest
class TestTaskEfficiencyEvaluatorBehavior:
    """Behavioral tests for the multi-turn (messages) path of TaskEfficiencyEvaluator."""

    def test_messages_valid_input_returns_expected_fields(self):
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)

        for key in (
            "task_efficiency",
            "task_efficiency_score",
            "task_efficiency_result",
            "task_efficiency_reason",
            "task_efficiency_status",
            "task_efficiency_threshold",
            "task_efficiency_passed",
            "task_efficiency_properties",
        ):
            assert key in result
        assert result["task_efficiency_status"] == "completed"
        assert isinstance(result["task_efficiency"], int)
        assert 1 <= result["task_efficiency"] <= 5

    def test_messages_with_tool_definitions(self):
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS, tool_definitions=VALID_TOOL_DEFINITIONS)

        assert result["task_efficiency_status"] == "completed"
        call_kwargs = evaluator._flow.call_args
        assert "tool_definitions" in call_kwargs.kwargs

    def test_messages_without_tool_definitions_does_not_pass_them(self):
        evaluator = _create_mocked_evaluator()
        evaluator(messages=VALID_MESSAGES_WITH_TOOLS)

        call_kwargs = evaluator._flow.call_args
        assert "tool_definitions" not in call_kwargs.kwargs

    def test_threshold_default_is_three(self):
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)
        assert result["task_efficiency_threshold"] == 3

    def test_threshold_override(self):
        evaluator = _create_mocked_evaluator(threshold=5)
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)
        # Mock returns GRADERS_SUCCESS_SCORE=5, so score=5 >= threshold=5 -> pass.
        assert result["task_efficiency_threshold"] == 5
        assert result["task_efficiency_passed"] is True
        assert result["task_efficiency_result"] == "pass"

    def test_threshold_above_score_fails(self):
        # Force a low score from the mock by patching the flow directly.
        evaluator = _create_mocked_evaluator(threshold=4)

        async def low_score_flow(timeout, **kwargs):
            return {
                "llm_output": {
                    "score": 2,
                    "reason": "Lots of repeated reads.",
                    "status": "completed",
                    "properties": {
                        "total_steps_count": 10,
                        "wasted_steps_count": 6,
                        "redundancy_findings": "auth.py read 4x",
                        "loop_findings": "None",
                    },
                }
            }

        evaluator._flow = MagicMock(side_effect=low_score_flow)
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)
        assert result["task_efficiency"] == 2
        assert result["task_efficiency_passed"] is False
        assert result["task_efficiency_result"] == "fail"

    def test_score_is_clamped_to_rubric_range(self):
        evaluator = _create_mocked_evaluator()

        async def out_of_range_flow(timeout, **kwargs):
            return {
                "llm_output": {
                    "score": 9,
                    "reason": "Out of range.",
                    "status": "completed",
                    "properties": {},
                }
            }

        evaluator._flow = MagicMock(side_effect=out_of_range_flow)
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)
        assert result["task_efficiency"] == 5

    def test_no_tool_calls_returns_skipped(self):
        """A text-only conversation has no trajectory; result is not_applicable / skipped."""
        evaluator = _create_mocked_evaluator()
        text_only_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hi."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        result = evaluator(messages=text_only_messages)

        assert result["task_efficiency"] is None
        assert result["task_efficiency_score"] is None
        assert result["task_efficiency_status"] == "skipped"
        assert result["task_efficiency_result"] == "not_applicable"
        assert result["task_efficiency_passed"] is None
        evaluator._flow.assert_not_called()

    def test_judge_skipped_status_propagates(self):
        evaluator = _create_mocked_evaluator()

        async def skipped_flow(timeout, **kwargs):
            return {
                "llm_output": {
                    "score": None,
                    "reason": "Not enough trajectory.",
                    "status": "skipped",
                    "properties": None,
                }
            }

        evaluator._flow = MagicMock(side_effect=skipped_flow)
        result = evaluator(messages=VALID_MESSAGES_WITH_TOOLS)
        assert result["task_efficiency_status"] == "skipped"
        assert result["task_efficiency"] is None

    def test_messages_empty_list_raises_error(self):
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_messages_missing_raises_error(self):
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator()

    def test_messages_invalid_type_raises_error(self):
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages="not a list")

    def test_messages_non_dict_items_raises_error(self):
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=[{"role": "user", "content": "hi"}, "not a dict"])

    def test_messages_missing_role_raises_error(self):
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=[{"content": [{"type": "text", "text": "no role"}]}])

    def test_function_call_normalized_to_tool_call(self):
        """function_call content should be recognized as a tool-use trajectory."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Do X."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "tool_call_id": "call_1",
                        "name": "do_x",
                        "arguments": {},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "function_call_output", "function_call_output": "ok"}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done."}]},
        ]
        result = evaluator(messages=messages)
        assert result["task_efficiency_status"] == "completed"


@pytest.mark.unittest
class TestTaskEfficiencyHelpers:
    """Unit tests for helper functions in _task_efficiency."""

    def test_count_assistant_tool_calls_zero(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert _count_assistant_tool_calls(msgs) == 0

    def test_count_assistant_tool_calls_counts_tool_call_types(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "name": "a", "tool_call_id": "1", "arguments": {}},
                    {"type": "function_call", "name": "b", "tool_call_id": "2", "arguments": {}},
                    {"type": "openapi_call", "name": "c", "tool_call_id": "3", "arguments": {}},
                    {"type": "text", "text": "not a tool call"},
                ],
            }
        ]
        assert _count_assistant_tool_calls(msgs) == 3

    def test_count_assistant_tool_calls_ignores_tool_role(self):
        msgs = [
            {
                "role": "tool",
                "tool_call_id": "1",
                "content": [{"type": "tool_call", "name": "x", "tool_call_id": "1", "arguments": {}}],
            }
        ]
        assert _count_assistant_tool_calls(msgs) == 0

    def test_serialize_messages_empty(self):
        assert serialize_messages([]) == ""

    def test_serialize_messages_includes_user_and_agent_text(self):
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "world"}]},
        ]
        text = serialize_messages(msgs)
        assert "hello" in text
        assert "world" in text
