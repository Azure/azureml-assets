# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct unit tests for shared evaluator utility functions.

These exercise the message-processing and evaluation-level helper functions that are
used across the prompty-based evaluators. They are imported from a representative
evaluator (``groundedness``), which currently defines the full superset of utilities.
Tests cover both golden exact-output assertions (regression guard) and structural
invariants.
"""

import copy

import pytest

from azure.ai.evaluation._exceptions import EvaluationException, ErrorTarget

from ...builtin.groundedness.evaluator._groundedness import (
    EvaluationLevel,
    _drop_mcp_approval_messages,
    _is_intermediate_response,
    _merge_query_response_messages,
    _normalize_function_call_types,
    _preprocess_messages,
    _resolve_evaluation_level,
    _split_messages_at_latest_user,
    _wrap_string_messages,
    serialize_messages,
    simplify_messages,
)


@pytest.mark.unittest
class TestDropMcpApprovalMessages:
    """Unit tests for _drop_mcp_approval_messages."""

    def test_drops_approval_request_and_response(self):
        """Assistant mcp_approval_request and tool mcp_approval_response are removed."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "mcp_approval_request", "name": "x", "tool_call_id": "r1"}]},
            {"role": "tool", "content": [{"type": "mcp_approval_response", "mcp_approval_response": True}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
        result = _drop_mcp_approval_messages(messages)
        assert result == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]

    def test_keeps_messages_without_approval(self):
        """Messages without approval content are unchanged."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        ]
        assert _drop_mcp_approval_messages(messages) == messages

    def test_non_list_passthrough(self):
        """A non-list input is returned unchanged."""
        assert _drop_mcp_approval_messages("not a list") == "not a list"
        assert _drop_mcp_approval_messages(None) is None


@pytest.mark.unittest
class TestNormalizeFunctionCallTypes:
    """Unit tests for _normalize_function_call_types."""

    def test_function_call_becomes_tool_call(self):
        """function_call type is renamed to tool_call."""
        messages = [{"role": "assistant", "content": [{"type": "function_call", "name": "f", "arguments": {}}]}]
        result = _normalize_function_call_types(copy.deepcopy(messages))
        assert result[0]["content"][0]["type"] == "tool_call"

    def test_function_call_output_becomes_tool_result_and_renames_key(self):
        """function_call_output -> tool_result and the payload key is renamed."""
        messages = [{"role": "tool", "content": [{"type": "function_call_output", "function_call_output": {"r": 1}}]}]
        result = _normalize_function_call_types(copy.deepcopy(messages))
        item = result[0]["content"][0]
        assert item["type"] == "tool_result"
        assert item["tool_result"] == {"r": 1}
        assert "function_call_output" not in item

    def test_openapi_call_and_output_normalized(self):
        """openapi_call -> tool_call and openapi_call_output -> tool_result with key rename."""
        messages = [
            {"role": "assistant", "content": [{"type": "openapi_call", "name": "o", "arguments": {}}]},
            {"role": "tool", "content": [{"type": "openapi_call_output", "openapi_call_output": {"r": 2}}]},
        ]
        result = _normalize_function_call_types(copy.deepcopy(messages))
        assert result[0]["content"][0]["type"] == "tool_call"
        assert result[1]["content"][0]["type"] == "tool_result"
        assert result[1]["content"][0]["tool_result"] == {"r": 2}
        assert "openapi_call_output" not in result[1]["content"][0]

    def test_non_list_passthrough(self):
        """A non-list input is returned unchanged."""
        assert _normalize_function_call_types("x") == "x"


@pytest.mark.unittest
class TestPreprocessMessages:
    """Unit tests for _preprocess_messages (drop approval + normalize types)."""

    def test_golden_combined_pipeline(self):
        """End-to-end golden output: approvals dropped, function/openapi types normalized."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "mcp_approval_request", "name": "x", "tool_call_id": "r1"}]},
            {"role": "tool", "content": [{"type": "mcp_approval_response", "mcp_approval_response": True}]},
            {"role": "assistant", "content": [{"type": "function_call", "tool_call_id": "c1", "name": "f",
                                               "arguments": {"a": 1}}]},
            {"role": "tool", "content": [{"type": "function_call_output", "function_call_output": {"r": 2}}]},
            {"role": "assistant", "content": [{"type": "openapi_call", "tool_call_id": "c2", "name": "o",
                                               "arguments": {}}]},
            {"role": "tool", "content": [{"type": "openapi_call_output", "openapi_call_output": {"r": 3}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]
        result = _preprocess_messages(copy.deepcopy(messages))
        assert result == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "c1", "name": "f",
                                               "arguments": {"a": 1}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_result": {"r": 2}}]},
            {"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "c2", "name": "o",
                                               "arguments": {}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_result": {"r": 3}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        ]

    def test_invariants_no_approval_or_function_types_remain(self):
        """Structural invariant: no approval/function/openapi content types survive."""
        messages = [
            {"role": "assistant", "content": [{"type": "mcp_approval_request", "name": "x"}]},
            {"role": "tool", "content": [{"type": "mcp_approval_response", "mcp_approval_response": True}]},
            {"role": "assistant", "content": [{"type": "function_call", "name": "f", "arguments": {}}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ]
        result = _preprocess_messages(copy.deepcopy(messages))
        seen_types = {c["type"] for m in result for c in m["content"]}
        assert seen_types.isdisjoint(
            {"mcp_approval_request", "mcp_approval_response", "function_call", "function_call_output",
             "openapi_call", "openapi_call_output"}
        )


@pytest.mark.unittest
class TestIsIntermediateResponse:
    """Unit tests for _is_intermediate_response."""

    def test_function_call_last_is_intermediate(self):
        """A response ending in a function_call is intermediate."""
        response = [{"role": "assistant", "content": [{"type": "function_call", "name": "f",
                                                       "tool_call_id": "c", "arguments": {}}]}]
        assert _is_intermediate_response(response) is True

    def test_mcp_approval_request_last_is_intermediate(self):
        """A response ending in mcp_approval_request is intermediate."""
        response = [{"role": "assistant", "content": [{"type": "mcp_approval_request", "name": "x"}]}]
        assert _is_intermediate_response(response) is True

    def test_text_last_is_not_intermediate(self):
        """A response ending in text is not intermediate."""
        response = [{"role": "assistant", "content": [{"type": "text", "text": "final"}]}]
        assert _is_intermediate_response(response) is False

    def test_non_list_or_empty_is_not_intermediate(self):
        """Non-list / empty / non-assistant inputs are not intermediate."""
        assert _is_intermediate_response(None) is False
        assert _is_intermediate_response([]) is False
        assert _is_intermediate_response("x") is False
        assert _is_intermediate_response([{"role": "user", "content": [{"type": "text", "text": "q"}]}]) is False


@pytest.mark.unittest
class TestSimplifyMessages:
    """Unit tests for simplify_messages."""

    def test_drops_system_and_simplifies_user_assistant(self):
        """System dropped by default; user/assistant text extracted into lists."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ans"}]},
        ]
        assert simplify_messages(messages) == [
            {"role": "user", "content": ["q"]},
            {"role": "assistant", "content": ["ans"]},
        ]

    def test_drop_tool_calls_removes_tool_and_toolcall_only_assistant(self):
        """With drop_tool_calls=True, tool messages and tool-call-only assistant messages are removed."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant",
             "content": [{"type": "tool_call", "name": "t", "tool_call_id": "c", "arguments": {}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_result": "r"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "ans"}]},
        ]
        assert simplify_messages(messages, drop_tool_calls=True) == [
            {"role": "user", "content": ["q"]},
            {"role": "assistant", "content": ["ans"]},
        ]

    def test_keep_system_when_drop_system_false(self):
        """drop_system=False retains the system message untouched."""
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}]},
        ]
        result = simplify_messages(messages, drop_system=False)
        assert result[0] == {"role": "system", "content": "sys"}

    def test_string_and_non_list_passthrough(self):
        """String / non-list inputs returned unchanged."""
        assert simplify_messages("a string") == "a string"
        assert simplify_messages(123) == 123


@pytest.mark.unittest
class TestSerializeMessages:
    """Golden-output unit tests for serialize_messages."""

    def test_empty_returns_empty_string(self):
        """No messages -> empty string."""
        assert serialize_messages([]) == ""

    def test_golden_system_query_toolcall_response(self):
        """Exact transcript for system + user + tool_call/tool_result + final text."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": [{"type": "text", "text": "Weather in Paris?"}]},
            {"role": "assistant", "content": [{"type": "tool_call", "tool_call_id": "c1", "name": "get_weather",
                                               "arguments": {"city": "Paris"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [{"type": "tool_result", "tool_result": "Sunny 22C"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It is sunny, 22C."}]},
        ]
        expected = (
            "SYSTEM_PROMPT:\n"
            "  You are helpful.\n"
            "\n"
            "User turn 1:\n"
            "  Weather in Paris?\n"
            "\n"
            "Agent turn 1:\n"
            '  [TOOL_CALL] get_weather(city="Paris")\n'
            "  [TOOL_RESULT] Sunny 22C\n"
            "  It is sunny, 22C."
        )
        assert serialize_messages(messages) == expected

    def test_string_content_preserved(self):
        """String content is serialized (not dropped)."""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]
        result = serialize_messages(messages)
        assert "What is 2+2?" in result
        assert "4" in result


@pytest.mark.unittest
class TestEvaluationLevelUtils:
    """Unit tests for evaluation-level helpers."""

    def test_resolve_none_and_empty_return_none(self):
        """None / empty string resolve to None (auto-detect)."""
        assert _resolve_evaluation_level(None, ErrorTarget.GROUNDEDNESS_EVALUATOR) is None
        assert _resolve_evaluation_level("", ErrorTarget.GROUNDEDNESS_EVALUATOR) is None

    def test_resolve_string_and_enum(self):
        """Valid string and enum inputs resolve to the matching member."""
        assert _resolve_evaluation_level("conversation", ErrorTarget.GROUNDEDNESS_EVALUATOR) == \
            EvaluationLevel.CONVERSATION
        assert _resolve_evaluation_level(EvaluationLevel.TURN, ErrorTarget.GROUNDEDNESS_EVALUATOR) == \
            EvaluationLevel.TURN

    def test_resolve_invalid_raises(self):
        """An invalid level string raises EvaluationException."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _resolve_evaluation_level("bogus", ErrorTarget.GROUNDEDNESS_EVALUATOR)

    def test_merge_query_response(self):
        """Merge concatenates query then response."""
        assert _merge_query_response_messages(
            [{"role": "user", "content": "q"}], [{"role": "assistant", "content": "r"}]
        ) == [{"role": "user", "content": "q"}, {"role": "assistant", "content": "r"}]

    def test_split_at_latest_user(self):
        """Split slices at the latest user turn (inclusive in query)."""
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        query, response = _split_messages_at_latest_user(messages)
        assert query == [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert response == [{"role": "assistant", "content": "d"}]

    def test_wrap_string_messages(self):
        """Wrap turns plain strings into role-tagged text content messages."""
        query, response = _wrap_string_messages("q", "r")
        assert query == [{"role": "user", "content": [{"type": "text", "text": "q"}]}]
        assert response == [{"role": "assistant", "content": [{"type": "text", "text": "r"}]}]
