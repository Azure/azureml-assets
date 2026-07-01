# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Output Utilization Evaluator."""

import json

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from .base_validator_unit_test import BaseValidatorUnitTest
from . import common_tool_test_data as data
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import (
    ConversationValidator,
    ToolDefinitionsValidator,
    ToolOutputUtilizationEvaluator,
    _get_agent_response,
    _stringify_tool_result,
)
# ErrorTarget is rebuilt by the module at import time so it carries the
# evaluator-specific TOOL_OUTPUT_UTILIZATION_EVALUATOR member.
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import (  # noqa: E402
    ErrorTarget,
)


@pytest.mark.unittest
class TestToolOutputUtilizationEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Output Utilization Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    # TOU defines its own _get_agent_response that JSON-encodes dict/list tool_result
    # payloads via _stringify_tool_result (vs the SDK helper used by TA/TC which uses
    # Python repr). The three fixtures below therefore use the TOU-flavored response
    # constants.
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.KB_MCP_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "response": data.MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MCP_TOU_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    # Phase 2: azure_ai_search, sharepoint_grounding, and azure_fabric are now
    # accepted by the TOU validator. The base class branches to NOT_APPLICABLE for
    # these tests whenever check_for_unsupported_tools is True, so we override the
    # three tests below to assert PASS instead.
    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.AZURE_AI_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.SHAREPOINT_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_TOU_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FABRIC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }
    # endregion

    evaluator_type = ToolOutputUtilizationEvaluator

    check_for_unsupported_tools = True

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.VALID_RESPONSE
    requires_tool_definitions = True

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()

    # --- Phase 2 overrides: these three tools used to be unsupported but TOU now
    # accepts them, so we override the base-class tests (which still branch to
    # NOT_APPLICABLE when check_for_unsupported_tools is True) to assert PASS.

    def test_azure_ai_search(self):
        """Azure AI Search tool with azure_ai_search type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="Azure AI Search",
            evaluation_inputs={
                "query": data.AZURE_AI_SEARCH_QUERY,
                "response": data.AZURE_AI_SEARCH_RESPONSE,
                "tool_definitions": data.AZURE_AI_SEARCH_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_azure_ai_search_expected_flow_inputs,
        )

    def test_sharepoint_grounding(self):
        """Test SharePoint grounding tool with sharepoint_grounding type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="SharePoint Grounding",
            evaluation_inputs={
                "query": data.SHAREPOINT_QUERY,
                "response": data.SHAREPOINT_RESPONSE,
                "tool_definitions": data.SHAREPOINT_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_sharepoint_grounding_expected_flow_inputs,
        )

    def test_fabric_data_agent(self):
        """Fabric data agent tool with azure_fabric type - now supported in Phase 2."""
        self._run_tool_type_test(
            test_label="Fabric Data Agent",
            evaluation_inputs={
                "query": data.FABRIC_QUERY,
                "response": data.FABRIC_RESPONSE,
                "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
            },
            assert_type=self.AssertType.PASS,
            expected_flow_inputs=self.test_fabric_data_agent_expected_flow_inputs,
        )


# ---------------------------------------------------------------------------
# Unit tests (moved from builtin/tests/unit/) for Phase 2 restricted-tool
# enablement: validator behavior + tool_result formatting.
# ---------------------------------------------------------------------------

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


def _make_tou_unit_eval_input(tool_name):
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


@pytest.mark.unittest
class TestUnsupportedToolsList:
    """The hard-coded UNSUPPORTED_TOOLS list controls service-side gating."""

    def test_newly_enabled_tools_are_not_in_unsupported_list(self):
        """Newly enabled tools must be absent from the unsupported list."""
        for tool_name in NEWLY_ENABLED_TOOLS:
            assert tool_name not in ConversationValidator.UNSUPPORTED_TOOLS

    def test_still_unsupported_tools_remain_in_list(self):
        """Still-unsupported tools must remain in the unsupported list."""
        for tool_name in STILL_UNSUPPORTED_TOOLS:
            assert tool_name in ConversationValidator.UNSUPPORTED_TOOLS

    def test_unsupported_list_contains_no_unexpected_tools(self):
        """Unsupported list must match the expected set exactly."""
        assert set(ConversationValidator.UNSUPPORTED_TOOLS) == set(STILL_UNSUPPORTED_TOOLS)


@pytest.mark.unittest
class TestValidatorAcceptsNewlyEnabledTools:
    """Verify SP / AAIS / Fabric tool calls now pass validation."""

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_assistant_message_accepts_tool(self, tool_name):
        """Assistant-message validation accepts each newly enabled tool."""
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
            check_for_unsupported_tools=True,
        )
        result = validator._validate_assistant_message(_make_tou_unit_eval_input(tool_name)["response"][0])
        assert result is None

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_validate_eval_input_accepts_tool(self, tool_name):
        """Full eval-input validation accepts each newly enabled tool."""
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
            check_for_unsupported_tools=True,
        )
        assert validator.validate_eval_input(_make_tou_unit_eval_input(tool_name)) is True


@pytest.mark.unittest
class TestValidatorRejectsStillUnsupportedTools:
    """The narrowing must not lift restrictions on the remaining tools."""

    @pytest.mark.parametrize("tool_name", STILL_UNSUPPORTED_TOOLS)
    def test_validate_eval_input_rejects_tool(self, tool_name):
        """Eval-input validation still rejects each still-unsupported tool."""
        validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            requires_query=True,
            optional_tool_definitions=False,
            check_for_unsupported_tools=True,
        )
        with pytest.raises(EvaluationException) as exc_info:
            validator.validate_eval_input(_make_tou_unit_eval_input(tool_name))
        assert "currently not supported" in str(exc_info.value)


@pytest.mark.unittest
class TestStringifyToolResult:
    """The new helper makes structured tool outputs LLM-readable."""

    def test_string_passes_through_unchanged(self):
        """String results pass through the helper unchanged."""
        assert _stringify_tool_result("hello") == "hello"

    def test_none_renders_as_empty_string(self):
        """None results render as the empty string (not 'None')."""
        assert _stringify_tool_result(None) == ""

    def test_dict_is_json_encoded(self):
        """Dict results are JSON-encoded."""
        result = _stringify_tool_result({"answer": 42, "ok": True})
        assert json.loads(result) == {"answer": 42, "ok": True}

    def test_list_of_dicts_is_json_encoded(self):
        """List-of-dicts results are JSON-encoded."""
        payload = [{"a": 1}, {"b": 2}]
        assert json.loads(_stringify_tool_result(payload)) == payload

    def test_unicode_is_preserved_not_escaped(self):
        """Unicode characters are preserved (ensure_ascii=False)."""
        rendered = _stringify_tool_result({"title": "测试"})
        assert "测试" in rendered


@pytest.mark.unittest
class TestGetAgentResponseFormatting:
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
        out = _get_agent_response(msgs, include_tool_messages=True)
        assert '[TOOL_RESULT] {"temp_f": 58}' in out

    def test_sharepoint_structured_result_is_json(self):
        """Structured SharePoint payloads render as valid JSON."""
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
        rendered_json = result_lines[0][len("[TOOL_RESULT] "):]
        assert json.loads(rendered_json) == sharepoint_payload
        # Python repr would emit single quotes — JSON must not.
        assert "'" not in rendered_json

    def test_azure_ai_search_dict_result_is_json(self):
        """Azure AI Search dict payloads render as valid JSON."""
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
        assert json.loads(result_lines[0][len("[TOOL_RESULT] "):]) == aas_payload

    def test_none_result_renders_empty(self):
        """None tool_result renders as an empty [TOOL_RESULT] body."""
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


@pytest.mark.unittest
class TestRealWorldSharePointTrace:
    """End-to-end smoke test using a payload shaped like a real Foundry playground OTel trace.

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
        """Verify a dict payload (the common ACA-parsed shape) round-trips as JSON.

        ACA usually parses the upstream response into a dict before the
        evaluator sees it. The helper must JSON-encode it so the judge
        can read it directly.
        """
        msgs = self._build_messages(self._SHAREPOINT_PAYLOAD)
        out = _get_agent_response(msgs, include_tool_messages=True)

        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        assert len(result_lines) == 1
        body = result_lines[0][len("[TOOL_RESULT] "):]

        parsed = json.loads(body)
        assert parsed == self._SHAREPOINT_PAYLOAD
        # JSON output never emits Python's single-quoted strings.
        assert "'" not in body
        # Distinctive structural content must survive intact for the judge.
        assert "knowledgeSourceIndex" in body
        assert "IT Onboarding Guide" in body

    def test_sharepoint_json_string_payload_passes_through(self):
        """Verify a raw JSON-string payload from the upstream passes through unchanged.

        The helper's str pass-through must leave it verbatim (no
        double-encoding), and json.loads must still work.
        """
        raw_json = json.dumps(self._SHAREPOINT_PAYLOAD)
        msgs = self._build_messages(raw_json)
        out = _get_agent_response(msgs, include_tool_messages=True)

        result_lines = [line for line in out if line.startswith("[TOOL_RESULT] ")]
        body = result_lines[0][len("[TOOL_RESULT] "):]
        assert body == raw_json
        assert json.loads(body) == self._SHAREPOINT_PAYLOAD


@pytest.mark.unittest
class TestToolOutputUtilizationValidatorUnit(BaseValidatorUnitTest):
    """Low-level unit tests for tool_output_utilization's repeated validators, utils and methods."""

    evaluator_class = ToolOutputUtilizationEvaluator
