# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ToolCallSuccessEvaluator.

These tests cover the changes that enable Azure AI Search, Microsoft
Fabric, and SharePoint grounding tool calls for the evaluator:

- The previously restricted tools ``azure_ai_search``, ``azure_fabric``,
  and ``sharepoint_grounding`` are accepted by the validator.
- The remaining restricted tools (Bing variants, web_search,
  browser_automation, code_interpreter_call, computer_call,
  openapi_call) are still rejected.
- The ``_get_tool_calls_results`` formatter renders structured
  tool_result payloads (lists / dicts produced by Azure AI Search,
  Fabric, SharePoint) as JSON so the LLM judge can parse them, while
  preserving the existing string pass-through for function / MCP
  tools.
"""

import json
import sys
from pathlib import Path

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

# Add the evaluator path to sys.path so we can import the in-tree module.
EVALUATOR_PATH = Path(__file__).parent.parent.parent / "tool_call_success" / "evaluator"
sys.path.insert(0, str(EVALUATOR_PATH))

from _tool_call_success import (  # noqa: E402
    ConversationValidator,
    ExtendedErrorTarget,
    ToolDefinitionsValidator,
    _get_tool_calls_results,
    _stringify_tool_result,
)

NEWLY_ENABLED_TOOLS = [
    "azure_ai_search",
    "azure_fabric",
    "sharepoint_grounding",
]

STILL_UNSUPPORTED_TOOLS = [
    "bing_grounding",
    "bing_custom_search",
    "browser_automation",
    "code_interpreter_call",
    "computer_call",
    "openapi_call",
    "web_search",
]


def _make_eval_input(tool_name: str, *, query=None):
    """Build a minimal eval_input dict with one assistant tool_call."""
    eval_input = {
        "response": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": tool_name,
                        "arguments": {"input": "anything"},
                    }
                ],
            }
        ],
    }
    if query is not None:
        eval_input["query"] = query
    return eval_input


class TestUnsupportedToolsList:
    """The hard-coded UNSUPPORTED_TOOLS list controls service-side gating."""

    def test_newly_enabled_tools_are_not_in_unsupported_list(self):
        for tool_name in NEWLY_ENABLED_TOOLS:
            assert tool_name not in ConversationValidator.UNSUPPORTED_TOOLS

    def test_still_unsupported_tools_remain_in_list(self):
        for tool_name in STILL_UNSUPPORTED_TOOLS:
            assert tool_name in ConversationValidator.UNSUPPORTED_TOOLS

    def test_unsupported_list_contains_no_unexpected_tools(self):
        # Defensive: keep the list explicit so future additions are
        # reviewed against this test.
        assert set(ConversationValidator.UNSUPPORTED_TOOLS) == set(STILL_UNSUPPORTED_TOOLS)


class TestValidatorAcceptsNewlyEnabledTools:
    """Verify SP / AAIS / Fabric tool calls now pass validation."""

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_assistant_message_accepts_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        # _validate_assistant_message returns None on success.
        result = validator._validate_assistant_message(_make_eval_input(tool_name)["response"][0])
        assert result is None

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_validate_eval_input_accepts_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        assert validator.validate_eval_input(_make_eval_input(tool_name)) is True


class TestValidatorRejectsStillUnsupportedTools:
    """The narrowing must not lift restrictions on the remaining tools."""

    @pytest.mark.parametrize("tool_name", STILL_UNSUPPORTED_TOOLS)
    def test_validate_eval_input_rejects_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        with pytest.raises(EvaluationException) as exc_info:
            validator.validate_eval_input(_make_eval_input(tool_name))
        assert "currently not supported" in str(exc_info.value)


class TestStringifyToolResult:
    """The new helper makes structured tool outputs LLM-readable."""

    def test_string_passes_through_unchanged(self):
        assert _stringify_tool_result("hello") == "hello"

    def test_none_renders_as_empty_string(self):
        # Avoid leaking the literal text "None" into the prompt.
        assert _stringify_tool_result(None) == ""

    def test_dict_is_json_encoded(self):
        result = _stringify_tool_result({"answer": 42, "ok": True})
        assert json.loads(result) == {"answer": 42, "ok": True}

    def test_list_of_dicts_is_json_encoded(self):
        sharepoint_payload = [
            {
                "documents": [
                    {
                        "title": "Q4 Earnings",
                        "content": "Revenue grew 12% YoY.",
                        "url": "https://contoso.sharepoint.com/q4",
                    }
                ]
            }
        ]
        rendered = _stringify_tool_result(sharepoint_payload)
        parsed = json.loads(rendered)
        assert parsed == sharepoint_payload

    def test_unicode_is_preserved_not_escaped(self):
        # ``ensure_ascii=False`` keeps non-ASCII readable for the judge.
        rendered = _stringify_tool_result({"title": "测试"})
        assert "测试" in rendered

    def test_non_json_value_falls_back_to_str(self):
        class _Custom:
            def __str__(self):
                return "<custom>"

        rendered = _stringify_tool_result(_Custom())
        assert rendered == '"<custom>"' or rendered == "<custom>"


class TestGetToolCallsResultsFormatting:
    """End-to-end formatting through the public helper."""

    def test_function_tool_string_result_unchanged(self):
        """Regression: function-style string outputs must not change."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "get_weather",
                        "arguments": {"city": "Seattle"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": '{"temp_f": 58}',
                    }
                ],
            },
        ]
        out = _get_tool_calls_results(msgs)
        assert out == [
            '[TOOL_CALL] get_weather(city="Seattle")',
            '[TOOL_RESULT] {"temp_f": 58}',
        ]

    def test_sharepoint_structured_result_is_json(self):
        sharepoint_payload = [
            {
                "documents": [
                    {
                        "title": "Q4 Earnings",
                        "content": "Revenue grew 12% YoY.",
                        "url": "https://contoso.sharepoint.com/q4",
                    }
                ]
            }
        ]
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "sharepoint_grounding",
                        "arguments": {"input": "Q4"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": sharepoint_payload,
                    }
                ],
            },
        ]
        out = _get_tool_calls_results(msgs)
        assert out[0] == '[TOOL_CALL] sharepoint_grounding(input="Q4")'
        # The result line must be valid JSON, not a Python repr.
        assert out[1].startswith("[TOOL_RESULT] ")
        rendered_json = out[1][len("[TOOL_RESULT] ") :]
        assert json.loads(rendered_json) == sharepoint_payload
        # And specifically: no single quotes (which Python repr would emit).
        assert "'" not in rendered_json

    def test_azure_ai_search_dict_result_is_json(self):
        aas_payload = {
            "results": [
                {"title": "Doc A", "score": 0.91},
                {"title": "Doc B", "score": 0.74},
            ]
        }
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "azure_ai_search",
                        "arguments": {"input": "revenue"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": aas_payload,
                    }
                ],
            },
        ]
        out = _get_tool_calls_results(msgs)
        rendered_json = out[1][len("[TOOL_RESULT] ") :]
        assert json.loads(rendered_json) == aas_payload

    def test_none_result_renders_empty(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "azure_fabric",
                        "arguments": {"input": "q"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": None}],
            },
        ]
        out = _get_tool_calls_results(msgs)
        assert out[1] == "[TOOL_RESULT] "
