# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Call Success Evaluator."""

import asyncio
import json

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_evaluator_behavior_test import _TurnLevelUtilE2ETests
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    LogSafeSummaryUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ...builtin.tool_call_success.evaluator._tool_call_success import (
    ConversationValidator,
    ExtendedErrorTarget,
    ToolCallSuccessEvaluator,
    ToolDefinitionsValidator,
    _collect_failed_tool_calls,
    _filter_to_used_tools,
    _format_value,
    _get_tool_calls_results,
    _reformat_tool_calls_results,
    _stringify_tool_result,
)
from ..common.evaluator_mock_config import create_mocked_evaluator


@pytest.mark.unittest
class TestToolCallSuccessEvaluatorBehavior(
    BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest, _TurnLevelUtilE2ETests
):
    """
    Behavioral tests for Tool Call Success Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "response": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "response": data.FILE_SEARCH_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.FILE_SEARCH_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "response": data.IMAGE_GEN_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.IMAGE_GEN_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "response": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "response": data.KB_MCP_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.KB_MCP_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "response": data.MCP_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.MCP_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    # Phase 2: azure_ai_search, sharepoint_grounding, and azure_fabric are now
    # accepted by the TCS validator. The base class branches to NOT_APPLICABLE for
    # these tests whenever check_for_unsupported_tools is True, so we override the
    # three tests below to assert PASS instead.
    test_azure_ai_search_expected_flow_inputs = {
        "response": data.AZURE_AI_SEARCH_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.AZURE_AI_SEARCH_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.AZURE_AI_SEARCH_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "response": data.SHAREPOINT_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.SHAREPOINT_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.SHAREPOINT_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "response": data.FABRIC_TCS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": data.FABRIC_TCS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FABRIC_TCS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolCallSuccessEvaluator

    # Phase 1 shipped only the prompty-level [STATUS] pass-through (asset version 8).
    # Phase 2 (this PR) flips the validator for the non-Bing restricted tools
    # (azure_ai_search, azure_fabric, sharepoint_grounding) so those three are now
    # accepted; bing_grounding and bing_custom_search remain rejected by the converter.
    check_for_unsupported_tools = True

    # Test Configs
    requires_query = False

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_results_without_arguments

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()

    # --- Phase 2 overrides: these three tools used to be unsupported but TCS now
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


# region Python-side short-circuit unit tests
#
# These exercise the two helpers introduced when runtime-status short-circuit
# moved from the prompty rubric into Python:
#
#   * ``_collect_failed_tool_calls`` -- identifies failed tool names across the
#     supported content shapes; drives the deterministic fail result.
#   * ``_get_tool_calls_results``   -- formats LLM input on the success path
#     and must NOT forward ``[STATUS]`` annotations (back-compat with the
#     pre-pass-through wire format).


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


def _assistant_parallel_tool_calls(blocks):
    """Build a single assistant message that emits multiple tool_call blocks in one turn.

    ``blocks`` is a list of ``(tool_call_id, name, arguments, status)`` tuples.
    This is the modern Responses-API topology for parallel function-call
    invocation: multiple ``tool_call`` content blocks under one assistant
    message, in contrast to one assistant message per call.
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


# endregion


@pytest.mark.unittest
class TestCollectFailedToolCalls:
    """Unit tests for the ``_collect_failed_tool_calls`` helper."""

    def test_no_status_anywhere_returns_empty(self):
        """No status field anywhere returns an empty failed-tool list."""
        msgs = [
            _assistant_tool_call("c1", "fetch_weather", {"city": "Seattle"}),
            _tool_result("c1", "Sunny, 72F."),
        ]
        assert _collect_failed_tool_calls(msgs) == []

    def test_all_completed_returns_empty(self):
        """All blocks marked ``completed`` returns an empty failed-tool list."""
        msgs = [
            _assistant_tool_call(
                "c1", "fetch_weather", {"city": "Seattle"}, status="completed"
            ),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
        ]
        assert _collect_failed_tool_calls(msgs) == []

    def test_failed_status_on_tool_call_block(self):
        """``failed`` status on the assistant tool_call block flags the tool as failed."""
        msgs = [
            _assistant_tool_call(
                "c1", "send_email", {"to": "x@example.com"}, status="failed"
            ),
            _tool_result("c1", ""),
        ]
        assert _collect_failed_tool_calls(msgs) == ["send_email"]

    def test_failed_status_on_tool_result_block(self):
        """``failed`` status on the tool_result block flags the matched tool as failed."""
        msgs = [
            _assistant_tool_call("c1", "send_email", {"to": "x@example.com"}),
            _tool_result("c1", "", status="failed"),
        ]
        assert _collect_failed_tool_calls(msgs) == ["send_email"]

    def test_incomplete_status_is_treated_as_failure(self):
        """``incomplete`` runtime status is treated as a failure, same as ``failed``."""
        msgs = [
            _assistant_tool_call("c1", "long_running_query", {}, status="incomplete"),
        ]
        assert _collect_failed_tool_calls(msgs) == ["long_running_query"]

    def test_failed_on_both_call_and_result_dedupes_to_single_entry(self):
        """Failure annotated on both the call and result blocks dedupes to one entry."""
        msgs = [
            _assistant_tool_call(
                "c1", "send_email", {"to": "x@example.com"}, status="failed"
            ),
            _tool_result("c1", "", status="failed"),
        ]
        assert _collect_failed_tool_calls(msgs) == ["send_email"]

    def test_unknown_runtime_status_is_ignored(self):
        """Statuses outside {failed, incomplete} fall through to the LLM rubric."""
        # Only "failed" and "incomplete" trigger the short-circuit; anything else
        # (including "error", "cancelled", "rate_limited", ...) falls through to
        # the LLM rubric for payload-based judgment, preserving back-compat with
        # runtimes that emit non-standardized status values.
        msgs = [
            _assistant_tool_call("c1", "send_email", {}, status="error"),
            _tool_result("c1", "", status="cancelled"),
        ]
        assert _collect_failed_tool_calls(msgs) == []

    def test_parallel_calls_one_failed_returns_only_the_failed_name(self):
        """In a parallel-call turn, only the failing tool is reported."""
        msgs = [
            _assistant_parallel_tool_calls(
                [
                    ("c1", "fetch_weather", {"city": "Seattle"}, "completed"),
                    ("c2", "send_email", {"to": "x@example.com"}, "failed"),
                    ("c3", "lookup_user", {"id": "u42"}, "completed"),
                ]
            ),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
            _tool_result("c2", "", status="failed"),
            _tool_result("c3", {"user_id": "u42"}, status="completed"),
        ]
        assert _collect_failed_tool_calls(msgs) == ["send_email"]

    def test_multiple_distinct_failures_preserve_order_and_dedupe(self):
        """Multiple distinct failures are deduped while preserving discovery order."""
        msgs = [
            _assistant_parallel_tool_calls(
                [
                    ("c1", "send_email", {"to": "x"}, "failed"),
                    ("c2", "fetch_weather", {"city": "Seattle"}, None),
                    ("c3", "lookup_user", {"id": "u42"}, "incomplete"),
                ]
            ),
            _tool_result("c2", "Sunny", status="failed"),
            # c1's tool_result also fails -- must not double-list send_email.
            _tool_result("c1", "", status="failed"),
        ]
        # send_email is seen first (assistant pass), then fetch_weather is
        # discovered on the tool pass, then lookup_user is discovered on the
        # assistant pass before the tool pass runs -- final ordering follows
        # the failed_ids list (assistant-pass first, then tool-pass).
        result = _collect_failed_tool_calls(msgs)
        assert set(result) == {"send_email", "fetch_weather", "lookup_user"}
        # send_email and lookup_user are recorded during the assistant pass
        # before fetch_weather appears in the tool pass.
        assert result.index("send_email") < result.index("fetch_weather")
        assert result.index("lookup_user") < result.index("fetch_weather")

    def test_failed_call_without_id_falls_back_to_name(self):
        """A failed tool_call missing its id is labeled by the tool name fallback."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "name": "anon_tool",
                        "arguments": {},
                        "status": "failed",
                    }
                ],
            }
        ]
        assert _collect_failed_tool_calls(msgs) == ["anon_tool"]

    def test_failed_tool_result_without_assistant_call_uses_id_as_label(self):
        """A failed tool_result with no matching call falls back to its id as label."""
        msgs = [
            _tool_result("c1", "", status="failed"),
        ]
        assert _collect_failed_tool_calls(msgs) == ["c1"]

    def test_nested_function_shape_failed_status(self):
        """Nested ``tool_call.function.name`` shape is recognized as a failure."""
        # The "tool_call.function.name" shape is what _normalize_function_call_types
        # produces from OpenAI Responses-API function_call blocks.
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": "c1",
                            "function": {
                                "name": "send_email",
                                "arguments": {"to": "x"},
                            },
                        },
                        "status": "failed",
                    }
                ],
            }
        ]
        assert _collect_failed_tool_calls(msgs) == ["send_email"]

    def test_non_list_input_returns_empty(self):
        """Non-list inputs (None, dict, str) return an empty failed-tool list."""
        assert _collect_failed_tool_calls(None) == []
        assert _collect_failed_tool_calls({}) == []
        assert _collect_failed_tool_calls("not a list") == []

    def test_malformed_content_blocks_are_skipped_silently(self):
        """Malformed/non-dict content blocks are skipped without raising."""
        msgs = [
            {"role": "assistant", "content": [None, "string", {"type": "text"}]},
            {"role": "tool", "tool_call_id": "c1", "content": [None]},
        ]
        assert _collect_failed_tool_calls(msgs) == []


@pytest.mark.unittest
class TestGetToolCallsResultsNoStatusForward:
    """``_get_tool_calls_results`` must not forward ``[STATUS]`` to the LLM input.

    Runtime status drives the Python short-circuit; the LLM rubric is only
    invoked on the success path and so the formatted output is byte-identical
    to the pre-status-pass-through wire format regardless of whether the
    source blocks carry a ``status`` field.
    """

    def test_status_on_tool_call_is_not_appended(self):
        """Status on the tool_call block is not forwarded into the formatted LLM input."""
        msgs = [
            _assistant_tool_call(
                "c1", "send_email", {"to": "x@example.com"}, status="failed"
            ),
            _tool_result("c1", ""),
        ]
        lines = _get_tool_calls_results(msgs)
        assert lines == [
            '[TOOL_CALL] send_email(to="x@example.com")',
            "[TOOL_RESULT] ",
        ]

    def test_status_on_tool_result_is_not_appended(self):
        """Status on the tool_result block is not forwarded into the formatted LLM input."""
        msgs = [
            _assistant_tool_call("c1", "send_email", {"to": "x@example.com"}),
            _tool_result("c1", "", status="failed"),
        ]
        lines = _get_tool_calls_results(msgs)
        assert lines == [
            '[TOOL_CALL] send_email(to="x@example.com")',
            "[TOOL_RESULT] ",
        ]

    def test_completed_status_is_not_appended(self):
        """``completed`` status is not appended to the formatted LLM input."""
        msgs = [
            _assistant_tool_call(
                "c1", "fetch_weather", {"city": "Seattle"}, status="completed"
            ),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
        ]
        lines = _get_tool_calls_results(msgs)
        assert lines == [
            '[TOOL_CALL] fetch_weather(city="Seattle")',
            "[TOOL_RESULT] Sunny, 72F.",
        ]

    def test_absent_status_back_compat_unchanged(self):
        """Absent status preserves the pre-pass-through formatted output verbatim."""
        msgs = [
            _assistant_tool_call("c1", "fetch_weather", {"city": "Seattle"}),
            _tool_result("c1", "Sunny, 72F."),
        ]
        lines = _get_tool_calls_results(msgs)
        assert lines == [
            '[TOOL_CALL] fetch_weather(city="Seattle")',
            "[TOOL_RESULT] Sunny, 72F.",
        ]

    def test_parallel_tool_calls_in_one_message_no_status_in_output(self):
        """Parallel tool calls render without any ``[STATUS]`` suffix in the output."""
        msgs = [
            _assistant_parallel_tool_calls(
                [
                    ("c1", "fetch_weather", {"city": "Seattle"}, "completed"),
                    ("c2", "send_email", {"to": "x@example.com"}, "completed"),
                    ("c3", "lookup_user", {"id": "u42"}, "completed"),
                ]
            ),
            _tool_result("c1", "Sunny, 72F.", status="completed"),
            _tool_result("c2", "ok", status="completed"),
            _tool_result("c3", {"user_id": "u42"}, status="completed"),
        ]
        lines = _get_tool_calls_results(msgs)
        # Dict tool_result is rendered as JSON by _stringify_tool_result.
        assert lines == [
            '[TOOL_CALL] fetch_weather(city="Seattle")',
            "[TOOL_RESULT] Sunny, 72F.",
            '[TOOL_CALL] send_email(to="x@example.com")',
            "[TOOL_RESULT] ok",
            '[TOOL_CALL] lookup_user(id="u42")',
            '[TOOL_RESULT] {"user_id": "u42"}',
        ]


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


def _make_tcs_unit_eval_input(tool_name, *, query=None):
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
        # Defensive: keep the list explicit so future additions are
        # reviewed against this test.
        assert set(ConversationValidator.UNSUPPORTED_TOOLS) == set(STILL_UNSUPPORTED_TOOLS)


@pytest.mark.unittest
class TestValidatorAcceptsNewlyEnabledTools:
    """Verify SP / AAIS / Fabric tool calls now pass validation."""

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_assistant_message_accepts_tool(self, tool_name):
        """Assistant-message validation accepts each newly enabled tool."""
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        # _validate_assistant_message returns None on success.
        result = validator._validate_assistant_message(_make_tcs_unit_eval_input(tool_name)["response"][0])
        assert result is None

    @pytest.mark.parametrize("tool_name", NEWLY_ENABLED_TOOLS)
    def test_validate_eval_input_accepts_tool(self, tool_name):
        """Full eval-input validation accepts each newly enabled tool."""
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        assert validator.validate_eval_input(_make_tcs_unit_eval_input(tool_name)) is True


@pytest.mark.unittest
class TestValidatorRejectsStillUnsupportedTools:
    """The narrowing must not lift restrictions on the remaining tools."""

    @pytest.mark.parametrize("tool_name", STILL_UNSUPPORTED_TOOLS)
    def test_validate_eval_input_rejects_tool(self, tool_name):
        """Eval-input validation still rejects each still-unsupported tool."""
        validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            optional_tool_definitions=True,
            check_for_unsupported_tools=True,
        )
        with pytest.raises(EvaluationException) as exc_info:
            validator.validate_eval_input(_make_tcs_unit_eval_input(tool_name))
        assert "currently not supported" in str(exc_info.value)


@pytest.mark.unittest
class TestStringifyToolResult:
    """The new helper makes structured tool outputs LLM-readable."""

    def test_string_passes_through_unchanged(self):
        """String results pass through the helper unchanged."""
        assert _stringify_tool_result("hello") == "hello"

    def test_none_renders_as_empty_string(self):
        """None results render as the empty string (not 'None')."""
        # Avoid leaking the literal text "None" into the prompt.
        assert _stringify_tool_result(None) == ""

    def test_dict_is_json_encoded(self):
        """Dict results are JSON-encoded."""
        result = _stringify_tool_result({"answer": 42, "ok": True})
        assert json.loads(result) == {"answer": 42, "ok": True}

    def test_list_of_dicts_is_json_encoded(self):
        """List-of-dicts results are JSON-encoded."""
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
        """Unicode characters are preserved (ensure_ascii=False)."""
        # ``ensure_ascii=False`` keeps non-ASCII readable for the judge.
        rendered = _stringify_tool_result({"title": "测试"})
        assert "测试" in rendered

    def test_non_json_value_falls_back_to_str(self):
        """Non-JSON-serializable values fall back to str()."""
        class _Custom:
            def __str__(self):
                return "<custom>"

        rendered = _stringify_tool_result(_Custom())
        assert rendered == '"<custom>"' or rendered == "<custom>"


@pytest.mark.unittest
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
        out = _get_tool_calls_results(msgs)
        assert out[0] == '[TOOL_CALL] sharepoint_grounding(input="Q4")'
        # The result line must be valid JSON, not a Python repr.
        assert out[1].startswith("[TOOL_RESULT] ")
        rendered_json = out[1][len("[TOOL_RESULT] "):]
        assert json.loads(rendered_json) == sharepoint_payload
        # And specifically: no single quotes (which Python repr would emit).
        assert "'" not in rendered_json

    def test_azure_ai_search_dict_result_is_json(self):
        """Azure AI Search dict payloads render as valid JSON."""
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
        rendered_json = out[1][len("[TOOL_RESULT] "):]
        assert json.loads(rendered_json) == aas_payload

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
        out = _get_tool_calls_results(msgs)
        assert out[1] == "[TOOL_RESULT] "


@pytest.mark.unittest
class TestRealWorldSharePointTrace:
    """End-to-end smoke test using a payload shaped like a real Foundry playground OTel trace.

    Mirrors the ``gen_ai.input.messages`` shape produced by a Foundry
    agent invoking SharePoint grounding: an assistant tool_call with a
    ``query`` argument followed by a tool message whose payload is the
    documents envelope ``{"documents": [{id, content, filepath, title,
    url, knowledgeSourceIndex}, ...]}``.

    The point is to catch regressions where realistic content (markdown,
    HTML escapes, unicode, embedded JSON, long strings) would be
    mangled by ``f"{result}"`` rendering. With the new helper the result
    must round-trip as valid JSON and the original document content must
    be reachable in the rendered string.
    """

    # Realistic SharePoint document payload shaped like a production
    # Foundry trace's ``tool_call_response.response`` field. Kept short
    # and topic-neutral; only the envelope shape matters.
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
        ]

    def test_sharepoint_dict_payload_round_trips_as_json(self):
        """Verify a dict payload (the common ACA-parsed shape) round-trips as JSON.

        ACA usually parses the upstream response into a dict before the
        evaluator sees it. The helper must JSON-encode it so the judge
        can read it directly.
        """
        msgs = self._build_messages(self._SHAREPOINT_PAYLOAD)
        out = _get_tool_calls_results(msgs)

        # One [TOOL_CALL] + one [TOOL_RESULT] line.
        assert len(out) == 2
        assert out[0] == '[TOOL_CALL] sharepoint_grounding(query="onboarding")'

        rendered = out[1]
        assert rendered.startswith("[TOOL_RESULT] ")
        body = rendered[len("[TOOL_RESULT] "):]

        # The rendered payload must be valid JSON and round-trip cleanly,
        # which is the whole point of switching off ``f"{result}"``.
        parsed = json.loads(body)
        assert parsed == self._SHAREPOINT_PAYLOAD

        # JSON output never emits Python's single-quoted strings.
        assert "'" not in body
        # Distinctive structural fields must survive so the judge can
        # ground on the documents envelope.
        assert "knowledgeSourceIndex" in body
        assert "IT Onboarding Guide" in body

    def test_sharepoint_json_string_payload_passes_through(self):
        """Verify a raw JSON-string payload passes through unchanged.

        Some upstreams hand the evaluator the raw JSON-encoded string.
        The helper's str pass-through must leave it verbatim (no
        double-encoding), and json.loads must still work.
        """
        raw_json = json.dumps(self._SHAREPOINT_PAYLOAD)
        msgs = self._build_messages(raw_json)
        out = _get_tool_calls_results(msgs)

        body = out[1][len("[TOOL_RESULT] "):]
        # String inputs pass through unchanged — no extra quoting.
        assert body == raw_json
        # And it's still valid JSON the judge can parse.
        assert json.loads(body) == self._SHAREPOINT_PAYLOAD


@pytest.mark.unittest
class TestToolCallSuccessValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    LogSafeSummaryUnitTests,
):
    """Low-level unit tests for tool_call_success's repeated validators, utils and methods."""

    evaluator_class = ToolCallSuccessEvaluator


@pytest.mark.unittest
class TestToolCallSuccessDoEvalBranches:
    """Cover tool_call_success ``_do_eval`` branches not exercised by the shared mixins."""

    def test_missing_response_raises(self):
        """Raise when the response field is absent."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({}))

    def test_none_response_raises(self):
        """Raise when the response is explicitly None."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"response": None}))

    def test_invalid_response_type_raises(self):
        """Raise when the response is neither a list nor a string."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "q", "response": 123}))

    def test_string_response_is_passed_through(self):
        """Pass a string response directly to the judge without reformatting."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        result = asyncio.run(evaluator._do_eval({"query": "q", "response": "final text answer"}))
        assert f"{evaluator._result_key}_score" in result

    def test_failed_tool_call_short_circuits(self):
        """Short-circuit to a deterministic fail result when a tool reports a failed status."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        response = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call_id": "c1", "name": "search", "arguments": {}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": "boom", "status": "failed"}],
            },
        ]
        result = asyncio.run(evaluator._do_eval({"query": "q", "response": response}))
        assert result[f"{evaluator._result_key}_result"] == "fail"

    def test_llm_skipped_status_returns_not_applicable(self):
        """Return a not-applicable result when the judge reports a skipped status."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")

        async def _skipped_flow(**kwargs):
            return {"llm_output": {"status": "skipped", "reason": "not applicable"}}

        evaluator._flow = _skipped_flow
        result = asyncio.run(evaluator._do_eval({"query": "q", "response": "text"}))
        assert isinstance(result, dict)

    def test_list_response_with_tool_calls_success(self):
        """Evaluate a well-formed tool-call response through the judge success path."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        response = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call_id": "c1", "name": "search", "arguments": {"q": "x"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": "found"}],
            },
        ]
        tool_definitions = [
            {"name": "search", "parameters": {"properties": {"q": {"type": "string"}}}}
        ]
        result = asyncio.run(
            evaluator._do_eval(
                {"query": "q", "response": response, "tool_definitions": tool_definitions}
            )
        )
        assert f"{evaluator._result_key}_score" in result

    def test_list_query_is_preprocessed(self):
        """Preprocess a list-shaped query before invoking the judge."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")
        query = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        result = asyncio.run(
            evaluator._do_eval({"query": query, "response": "final text answer"})
        )
        assert f"{evaluator._result_key}_score" in result

    def test_non_dict_llm_output_raises(self):
        """Raise when the judge returns a non-dict ``llm_output`` payload."""
        evaluator = create_mocked_evaluator(ToolCallSuccessEvaluator, "tool_call_success")

        async def _bad_flow(**kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = _bad_flow
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "q", "response": "text"}))


@pytest.mark.unittest
class TestToolCallSuccessHelperBranches:
    """Cover module-level helper branches not reached through ``_do_eval``."""

    def test_format_value_none(self):
        """Render None as the literal string ``None``."""
        assert _format_value(None) == "None"

    def test_stringify_tool_result_falls_back_on_unserializable(self):
        """Fall back to ``str`` when JSON encoding raises."""
        circular = {}
        circular["self"] = circular
        rendered = _stringify_tool_result(circular)
        assert isinstance(rendered, str)

    def test_filter_to_used_tools_nested_function_shape(self):
        """Match used tools declared with the nested ``tool_call.function`` shape."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call": {"function": "search"}}
                ],
            }
        ]
        tool_definitions = [{"name": "search"}, {"name": "unused"}]
        filtered = _filter_to_used_tools(tool_definitions, msgs)
        assert filtered == [{"name": "search"}]

    def test_get_tool_calls_results_nested_function_shape(self):
        """Format tool calls declared with the nested ``tool_call.function`` object."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call": {
                            "id": "c1",
                            "function": {"name": "search", "arguments": {"q": "x"}},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": "found"}],
            },
        ]
        out = _get_tool_calls_results(msgs)
        assert out[0] == '[TOOL_CALL] search(q="x")'
        assert out[1] == "[TOOL_RESULT] found"

    def test_reformat_tool_calls_results_none_returns_empty(self):
        """Return an empty string when reformatting a ``None`` response."""
        assert _reformat_tool_calls_results(None) == ""
