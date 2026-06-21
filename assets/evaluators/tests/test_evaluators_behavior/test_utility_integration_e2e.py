# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""End-to-end (mocked, no LLM) tests for shared utilities inside the evaluator call path.

These verify that the message-processing utilities (``_preprocess_messages`` ->
``_drop_mcp_approval_messages`` + ``_normalize_function_call_types``, and
``serialize_messages``) run correctly when invoked through an evaluator's multi-turn
``__call__``. The evaluator's LLM flow is mocked, and the exact serialized transcript
handed to the flow is asserted against a golden value so that a regression in any of
those utilities is caught here.
"""

import os
from unittest.mock import MagicMock

import pytest

from azure.ai.evaluation import AzureOpenAIModelConfiguration

from ...builtin.coherence.evaluator._coherence import CoherenceEvaluator
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator


def _model_config():
    return AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )


def _multi_turn_side_effect(score):
    async def flow_side_effect(timeout, **kwargs):
        return {"llm_output": {"score": score, "reason": "ok", "status": "completed", "properties": {}}}

    return flow_side_effect


def _mocked(evaluator_cls, score):
    """Create an evaluator with both single- and multi-turn flows mocked."""
    evaluator = evaluator_cls(model_config=_model_config())
    evaluator._flow = MagicMock(side_effect=_multi_turn_side_effect(score))
    evaluator._multi_turn_flow = MagicMock(side_effect=_multi_turn_side_effect(score))
    return evaluator


# Messages exercising preprocessing: MCP approval (dropped) + function_call/function_call_output
# (normalized to tool_call/tool_result and rendered as [TOOL_CALL]/[TOOL_RESULT]).
GROUNDEDNESS_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Q1"}]},
    {"role": "assistant", "content": [{"type": "mcp_approval_request", "name": "x", "tool_call_id": "r1"}]},
    {"role": "tool", "content": [{"type": "mcp_approval_response", "mcp_approval_response": True}]},
    {"role": "assistant", "content": [{"type": "function_call", "tool_call_id": "c1", "name": "search",
                                       "arguments": {"q": "a"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": [{"type": "function_call_output",
                                                        "function_call_output": "RESULT"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "final answer"}]},
]

COHERENCE_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
    {"role": "assistant", "content": [{"type": "function_call", "tool_call_id": "c1", "name": "lookup",
                                       "arguments": {"k": "v"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": [{"type": "function_call_output",
                                                        "function_call_output": "DATA"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Here is the info."}]},
]

TASK_COMPLETION_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "Do task"}]},
    {"role": "assistant", "content": [{"type": "function_call", "tool_call_id": "c1", "name": "run",
                                       "arguments": {"x": 1}}]},
    {"role": "tool", "tool_call_id": "c1", "content": [{"type": "function_call_output",
                                                        "function_call_output": "OK"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Task done."}]},
]


@pytest.mark.unittest
class TestSerializeAndPreprocessE2E:
    """Verify serialize_messages + _preprocess_messages output via the evaluator call path."""

    def test_groundedness_multi_turn_serialized_transcript_golden(self):
        """Groundedness: MCP dropped, function types normalized, transcript matches golden."""
        evaluator = _mocked(GroundednessEvaluator, score=5)
        result = evaluator(messages=GROUNDEDNESS_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        transcript = evaluator._multi_turn_flow.call_args.kwargs["messages"]
        assert transcript == (
            "User turn 1:\n"
            "  Q1\n"
            "\n"
            "Agent turn 1:\n"
            '  [TOOL_CALL] search(q="a")\n'
            "  [TOOL_RESULT] RESULT\n"
            "  final answer"
        )
        # Preprocessing invariants
        assert "mcp_approval" not in transcript
        assert "function_call" not in transcript
        assert "[TOOL_CALL]" in transcript and "[TOOL_RESULT]" in transcript
        # Pipeline still produced a valid score
        assert result["groundedness_status"] == "completed"

    def test_coherence_multi_turn_serialized_transcript_golden(self):
        """Coherence: function_call/output normalized and rendered, transcript matches golden."""
        evaluator = _mocked(CoherenceEvaluator, score=4)
        evaluator(messages=COHERENCE_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        transcript = evaluator._multi_turn_flow.call_args.kwargs["messages"]
        assert transcript == (
            "User turn 1:\n"
            "  Hello\n"
            "\n"
            "Agent turn 1:\n"
            '  [TOOL_CALL] lookup(k="v")\n'
            "  [TOOL_RESULT] DATA\n"
            "  Here is the info."
        )

    def test_task_completion_multi_turn_serialized_transcript_golden(self):
        """Task completion: transcript matches golden, proving shared utilities are applied."""
        evaluator = _mocked(TaskCompletionEvaluator, score=1)
        evaluator(messages=TASK_COMPLETION_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        transcript = evaluator._multi_turn_flow.call_args.kwargs["messages"]
        assert transcript == (
            "User turn 1:\n"
            "  Do task\n"
            "\n"
            "Agent turn 1:\n"
            '  [TOOL_CALL] run(x="1")\n'
            "  [TOOL_RESULT] OK\n"
            "  Task done."
        )

    def test_multi_turn_path_does_not_call_single_turn_flow(self):
        """Messages input routes to the multi-turn flow only (single-turn flow untouched)."""
        evaluator = _mocked(GroundednessEvaluator, score=5)
        evaluator(messages=GROUNDEDNESS_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
