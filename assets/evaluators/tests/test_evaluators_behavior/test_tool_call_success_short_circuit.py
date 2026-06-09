# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ToolCallSuccess deterministic status-based short-circuit.

When the agent runtime reports a known-failure ``status`` on any tool_call /
tool_result content block (e.g. "failed", "error", "incomplete"), the
evaluator deterministically returns a ``fail`` result without calling the
LLM. Absent ``status``, behavior is unchanged.
"""

import pytest

from ...builtin.tool_call_success.evaluator._tool_call_success import (
    ToolCallSuccessEvaluator,
    _FAILED_TOOL_STATUSES,
    _collect_failed_tool_statuses,
)
from ..common.base_prompty_evaluator_runner import BasePromptyEvaluatorRunner


# region helpers


def _assistant_parallel_tool_calls(blocks):
    """Build a single assistant message that emits multiple tool_call blocks in one turn.

    ``blocks`` is a list of ``(tool_call_id, name, arguments, status)`` tuples.
    This shapes the message the way modern Responses-API runtimes emit parallel
    tool calls (multiple ``tool_call`` content blocks under one assistant message),
    in contrast to one assistant message per call.
    """
    content = []
    for tool_call_id, name, arguments, status in blocks:
        block = {
            "type": "tool_call",
            "tool_call_id": tool_call_id,
            "name": name,
            "arguments": arguments,
        }
        if status is not None:
            block["status"] = status
        content.append(block)
    return {"role": "assistant", "content": content}


def _assistant_tool_call(tool_call_id, name, arguments, status=None):
    """Build an assistant message carrying a single tool_call content block."""
    block = {
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        "name": name,
        "arguments": arguments,
    }
    if status is not None:
        block["status"] = status
    return {"role": "assistant", "content": [block]}


def _tool_result(tool_call_id, result, status=None):
    """Build a tool message carrying a single tool_result content block."""
    block = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "tool_result": result,
    }
    if status is not None:
        block["status"] = status
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": [block],
    }


def _failing_response():
    """Build a minimal agent response with a failed tool execution."""
    return [
        _assistant_tool_call("call_1", "fetch_weather", {"location": "Seattle"}, status="failed"),
        _tool_result("call_1", "", status="failed"),
    ]


# endregion


@pytest.mark.unittest
class TestCollectFailedToolStatuses:
    """Unit tests for the ``_collect_failed_tool_statuses`` helper."""

    @pytest.mark.parametrize("status", sorted(_FAILED_TOOL_STATUSES))
    def test_each_failure_status_is_detected(self, status):
        """Every status in ``_FAILED_TOOL_STATUSES`` is returned when seen on a tool_call."""
        msgs = [_assistant_tool_call("c1", "x", {}, status=status)]
        assert _collect_failed_tool_statuses(msgs) == [status]

    def test_case_insensitive_match(self):
        """Match is case-insensitive; the returned value is lowercased."""
        msgs = [_assistant_tool_call("c1", "x", {}, status="FAILED")]
        assert _collect_failed_tool_statuses(msgs) == ["failed"]

    def test_completed_status_is_not_detected(self):
        """Successful ``completed`` status is not treated as a failure signal."""
        msgs = [_assistant_tool_call("c1", "x", {}, status="completed")]
        assert _collect_failed_tool_statuses(msgs) == []

    def test_missing_status_is_not_detected(self):
        """Absent ``status`` field produces no failure signal (LLM path remains)."""
        msgs = [_assistant_tool_call("c1", "x", {})]
        assert _collect_failed_tool_statuses(msgs) == []

    def test_status_on_tool_result_is_detected(self):
        """Failure status on a ``tool_result`` block is detected, not just on tool_call."""
        msgs = [_tool_result("c1", "", status="error")]
        assert _collect_failed_tool_statuses(msgs) == ["error"]

    def test_duplicates_preserved_in_return(self):
        """Duplicates are preserved so callers can dedupe at their own granularity."""
        msgs = [
            _assistant_tool_call("c1", "x", {}, status="failed"),
            _tool_result("c1", "", status="failed"),
        ]
        assert _collect_failed_tool_statuses(msgs) == ["failed", "failed"]

    def test_status_on_unrelated_content_type_is_ignored(self):
        """Status fields on non-tool content blocks (e.g. ``text``) are ignored."""
        msgs = [{"role": "assistant", "content": [{"type": "text", "text": "hi", "status": "failed"}]}]
        assert _collect_failed_tool_statuses(msgs) == []

    def test_non_list_input_returns_empty(self):
        """Non-list inputs (None, string, dict) return empty without raising."""
        assert _collect_failed_tool_statuses(None) == []
        assert _collect_failed_tool_statuses("not-a-list") == []
        assert _collect_failed_tool_statuses({"role": "assistant"}) == []

    def test_malformed_messages_are_tolerated(self):
        """Malformed entries (None / non-dict / wrong-shape content) are skipped."""
        msgs = [
            None,
            "not-a-dict",
            {"role": "assistant"},
            {"role": "assistant", "content": "stringly"},
            {"role": "assistant", "content": [None, "x", {"type": "tool_call", "status": "failed"}]},
        ]
        assert _collect_failed_tool_statuses(msgs) == ["failed"]

    def test_unknown_status_string_is_ignored(self):
        """Unknown status strings outside ``_FAILED_TOOL_STATUSES`` are ignored."""
        msgs = [_assistant_tool_call("c1", "x", {}, status="weird_state")]
        assert _collect_failed_tool_statuses(msgs) == []


@pytest.mark.unittest
class TestToolCallSuccessShortCircuit(BasePromptyEvaluatorRunner):
    """Integration tests that the evaluator short-circuits before invoking the LLM."""

    evaluator_type = ToolCallSuccessEvaluator

    def _failing_query(self):
        """Return a minimal user query used by the short-circuit integration tests."""
        return [{"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]}]

    def test_short_circuit_when_tool_call_status_is_failed(self):
        """A ``failed`` status on a tool_call returns ``fail`` without invoking the LLM."""
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=self._failing_query(),
            response=_failing_response(),
        )
        assert results["tool_call_success_result"] == "fail"
        assert results["tool_call_success_passed"] is False
        assert results["tool_call_success_score"] == 0.0
        assert results["tool_call_success_status"] == "completed"
        properties = results["tool_call_success_properties"]
        assert properties["short_circuit"] == "tool_status"
        assert properties["failed_statuses"] == ["failed"]
        flow_mock.assert_not_called()

    def test_short_circuit_dedupes_failed_statuses_in_properties(self):
        """Properties surface a deduped + sorted set of failure statuses."""
        response = [
            _assistant_tool_call("c1", "fetch_weather", {"location": "Seattle"}, status="failed"),
            _tool_result("c1", "", status="error"),
            _assistant_tool_call("c2", "send_email", {"to": "x@example.com"}, status="failed"),
        ]
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=self._failing_query(),
            response=response,
        )
        properties = results["tool_call_success_properties"]
        assert properties["failed_statuses"] == ["error", "failed"]
        flow_mock.assert_not_called()

    def test_no_short_circuit_when_all_statuses_completed(self):
        """When all statuses are ``completed`` the LLM rubric path runs normally."""
        response = [
            _assistant_tool_call("c1", "fetch_weather", {"location": "Seattle"}, status="completed"),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
        ]
        _, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=self._failing_query(),
            response=response,
        )
        flow_mock.assert_called_once()

    def test_no_short_circuit_when_status_absent(self):
        """When ``status`` is absent the LLM rubric path runs normally (back-compat)."""
        response = [
            _assistant_tool_call("c1", "fetch_weather", {"location": "Seattle"}),
            _tool_result("c1", "Sunny, 72F."),
        ]
        _, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=self._failing_query(),
            response=response,
        )
        flow_mock.assert_called_once()

    def test_short_circuit_on_parallel_tool_calls_in_one_assistant_message(self):
        """Short-circuit fires when failed status is on one of N parallel calls in a single assistant message.

        The modern Responses-API topology emits multiple ``tool_call`` blocks
        under one assistant message's ``content`` list (parallel function-call
        invocation). The short-circuit must walk into that content list and
        flag a single failed block among otherwise-completed siblings -- not
        only when failed calls live in their own separate assistant messages
        (which the dedupe test already covers).
        """
        response = [
            _assistant_parallel_tool_calls([
                ("c1", "fetch_weather", {"city": "Seattle"}, "completed"),
                ("c2", "send_email",   {"to": "x@example.com"}, "failed"),
                ("c3", "lookup_user",  {"id": "u42"}, "completed"),
            ]),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
            _tool_result("c2", "", status="failed"),
            _tool_result("c3", {"user_id": "u42"}, status="completed"),
        ]
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=self._failing_query(),
            response=response,
        )
        assert results["tool_call_success_result"] == "fail"
        assert results["tool_call_success_passed"] is False
        assert results["tool_call_success_score"] == 0.0
        properties = results["tool_call_success_properties"]
        assert properties["short_circuit"] == "tool_status"
        # Only "failed" is in _FAILED_TOOL_STATUSES among these three; dedupes the
        # tool_call + tool_result pair carrying it.
        assert properties["failed_statuses"] == ["failed"]
        flow_mock.assert_not_called()
