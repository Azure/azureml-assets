# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the ToolOutputUtilizationEvaluator.

These tests cover the changes that enable Azure AI Search, Microsoft
Fabric, and SharePoint grounding tool calls for the evaluator:

- The previously restricted tools ``azure_ai_search``, ``azure_fabric``,
  and ``sharepoint_grounding`` are accepted by the validator.
- The remaining restricted tools (Bing variants, web_search,
  browser_automation, code_interpreter_call, computer_call,
  openapi_call) are still rejected.
- The ``_get_agent_response`` formatter renders structured
  tool_result payloads (lists / dicts produced by Azure AI Search,
  Fabric, SharePoint) as JSON so the LLM judge can parse them.
"""

import json
import sys
from pathlib import Path

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

EVALUATOR_PATH = Path(__file__).parent.parent.parent / "tool_output_utilization" / "evaluator"
sys.path.insert(0, str(EVALUATOR_PATH))

from _tool_output_utilization import (  # noqa: E402
    ConversationValidator,
    ToolDefinitionsValidator,
    _get_agent_response,
    _stringify_tool_result,
)

# ErrorTarget is rebuilt by the module at import time so it carries the
# evaluator-specific TOOL_OUTPUT_UTILIZATION_EVALUATOR member.
from _tool_output_utilization import ErrorTarget  # noqa: E402

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


def _make_eval_input(tool_name: str):
    """Build a minimal eval_input dict with one assistant tool_call.

    Tool-Output-Utilization requires a query input.
    """
    return {
        "query": [{"role": "user", "content": [{"type": "text", "text": "Find docs"}]}],
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
        "tool_definitions": [
            {
                "name": tool_name,
                "description": f"{tool_name} description",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                },
            }
        ],
    }


class TestUnsupportedToolsList:
    """The hard-coded UNSUPPORTED_TOOLS list controls service-side gating."""

    def test_newly_enabled_tools_are_not_in_unsupported_list(self):
        for tool_name in NEWLY_ENABLED_TOOLS:
            assert tool_name not in ConversationValidator.UNSUPPORTED_TOOLS

    def test_still_unsupported_tools_remain_in_list(self):
        for tool_name in STILL_UNSUPPORTED_TOOLS:
            assert tool_name in ConversationValidator.UNSUPPORTED_TOOLS

    def test_unsupported_list_contains_no_unexpected_tools(self):
        assert set(ConversationValidator.UNSUPPORTED_TOOLS) == set(STILL_UNSUPPORTED_TOOLS)


class TestValidatorAcceptsNewlyEnabledTools:
    """Verify SP / AAIS / Fabric tool calls now pass validation."""

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_assistant_message_accepts_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
            check_for_unsupported_tools=True,
        )
        result = validator._validate_assistant_message(_make_eval_input(tool_name)["response"][0])
        assert result is None

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_validate_eval_input_accepts_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
            check_for_unsupported_tools=True,
        )
        assert validator.validate_eval_input(_make_eval_input(tool_name)) is True


class TestValidatorRejectsStillUnsupportedTools:
    """The narrowing must not lift restrictions on the remaining tools."""

    @pytest.mark.parametrize("tool_name", STILL_UNSUPPORTED_TOOLS)
    def test_validate_eval_input_rejects_tool(self, tool_name):
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
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
        assert _stringify_tool_result(None) == ""

    def test_dict_is_json_encoded(self):
        result = _stringify_tool_result({"answer": 42, "ok": True})
        assert json.loads(result) == {"answer": 42, "ok": True}

    def test_list_of_dicts_is_json_encoded(self):
        payload = [{"a": 1}, {"b": 2}]
        assert json.loads(_stringify_tool_result(payload)) == payload

    def test_unicode_is_preserved_not_escaped(self):
        rendered = _stringify_tool_result({"title": "测试"})
        assert "测试" in rendered


class TestGetAgentResponseFormatting:
    """End-to-end formatting through the public helper."""

    def test_function_tool_string_result_unchanged(self):
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
        out = _get_agent_response(msgs, include_tool_messages=True)
        assert '[TOOL_RESULT] {"temp_f": 58}' in out

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
        out = _get_agent_response(msgs, include_tool_messages=True)
        # Find the tool_result line
        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        assert len(result_lines) == 1
        rendered_json = result_lines[0][len("[TOOL_RESULT] ") :]
        assert json.loads(rendered_json) == sharepoint_payload
        # Python repr would emit single quotes — JSON must not.
        assert "'" not in rendered_json

    def test_azure_ai_search_dict_result_is_json(self):
        aas_payload = {"results": [{"title": "Doc A", "score": 0.91}]}
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
        out = _get_agent_response(msgs, include_tool_messages=True)
        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        assert json.loads(result_lines[0][len("[TOOL_RESULT] ") :]) == aas_payload

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
        out = _get_agent_response(msgs, include_tool_messages=True)
        assert "[TOOL_RESULT] " in out

    def test_include_tool_messages_false_omits_tool_results(self):
        """Regression: turning off tool message inclusion still works."""
        msgs = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi"}],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": {"data": 1},
                    }
                ],
            },
        ]
        out = _get_agent_response(msgs, include_tool_messages=False)
        assert all("[TOOL_RESULT]" not in line for line in out)


class TestRealWorldSharePointTrace:
    """End-to-end smoke test using a payload shaped like a real
    Foundry playground OTel trace.

    Mirrors the ``gen_ai.input.messages`` shape produced by a Foundry
    agent invoking SharePoint grounding: an assistant tool_call with a
    ``query`` argument followed by a tool message whose payload is the
    documents envelope ``{"documents": [{id, content, filepath, title,
    url, knowledgeSourceIndex}, ...]}``.

    Tool Output Utilization specifically asks the judge "did the agent
    use the tool output?", so the rendered TOOL_RESULT must preserve
    the document content verbatim (no Python repr mangling) for the
    judge to make that call.
    """

    _SHAREPOINT_PAYLOAD = {
        "documents": [
            {
                "id": "0",
                "content": "Onboarding doc: see the setup guide for first-time access.",
                "filepath": "https://contoso.sharepoint.com/sites/it/SitePages/Onboarding.aspx",
                "title": "IT Onboarding Guide",
                "url": "https://contoso.sharepoint.com/sites/it/SitePages/Onboarding.aspx",
                "knowledgeSourceIndex": 0,
            }
        ]
    }

    def _build_messages(self, tool_result_value):
        """Construct messages mirroring the production trace shape."""
        return [
            {
                "role": "user",
                "content": [{"type": "text", "text": "how do I get started?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_sp_1",
                        "name": "sharepoint_grounding",
                        "arguments": {"query": "onboarding"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_sp_1",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": tool_result_value,
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "To get started, follow the IT Onboarding Guide.",
                    }
                ],
            },
        ]

    def test_sharepoint_dict_payload_round_trips_as_json(self):
        """Most common shape: ACA parses the response into a dict
        before the evaluator sees it. The helper must JSON-encode it."""
        msgs = self._build_messages(self._SHAREPOINT_PAYLOAD)
        out = _get_agent_response(msgs, include_tool_messages=True)

        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        assert len(result_lines) == 1
        body = result_lines[0][len("[TOOL_RESULT] ") :]

        parsed = json.loads(body)
        assert parsed == self._SHAREPOINT_PAYLOAD
        # JSON output never emits Python's single-quoted strings.
        assert "'" not in body
        # Distinctive structural content must survive intact for the judge.
        assert "knowledgeSourceIndex" in body
        assert "IT Onboarding Guide" in body

    def test_sharepoint_json_string_payload_passes_through(self):
        """Alternate shape: raw JSON-encoded string from the upstream.
        The helper's str pass-through must leave it verbatim."""
        raw_json = json.dumps(self._SHAREPOINT_PAYLOAD)
        msgs = self._build_messages(raw_json)
        out = _get_agent_response(msgs, include_tool_messages=True)

        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        body = result_lines[0][len("[TOOL_RESULT] ") :]
        assert body == raw_json
        assert json.loads(body) == self._SHAREPOINT_PAYLOAD
