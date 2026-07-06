# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Selection Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import asyncio

import pytest

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    ConversationHistoryReformatUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    LogSafeSummaryUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionExtractionUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ...builtin.tool_selection.evaluator._tool_selection import (
    ToolSelectionEvaluator,
    _extract_tool_names_from_calls,
    _format_value,
    _get_built_in_tool_definition,
    _get_conversation_history,
    _log_safe_summary,
)
from ..common.evaluator_mock_config import create_mocked_evaluator


@pytest.mark.unittest
class TestToolSelectionEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Selection Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.LOCAL_CALLS_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "tool_calls": data.CODE_INTERPRETER_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.CODE_INTERPRETER_TOOL_DEFINITIONS,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BING_GROUNDING_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BING_GROUNDING_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BING_CUSTOM_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BING_CUSTOM_SEARCH_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FILE_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.AZURE_AI_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.AZURE_AI_SEARCH_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "tool_calls": data.SHAREPOINT_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.SHAREPOINT_TS_EXPECTED_FLOW_TOOL_DEFINITIONS,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "tool_calls": data.FABRIC_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FABRIC_TOOL_DEFINITIONS,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "tool_calls": ["weather_GetCurrentWeather"],
        "tool_definitions": data.OPENAPI_TOOL_DEFINITIONS,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.WEB_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.WEB_SEARCH_TOOL_DEFINITIONS,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_TS_EXPECTED_FLOW_QUERY,
        "tool_calls": data.BROWSER_AUTOMATION_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.BROWSER_AUTOMATION_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "tool_calls": data.IMAGE_GEN_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MEMORY_SEARCH_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.KB_MCP_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "tool_calls": data.MCP_TS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    is_tool_definition_required = True

    evaluator_type = ToolSelectionEvaluator

    def test_skipped_llm_status_returns_not_applicable(self):
        """Flow output with status='skipped' yields a not-applicable result, not a crash."""
        self.run_skipped_llm_status_not_applicable_test()

    def test_intermediate_response_returns_not_applicable(self):
        """A response ending in an unresolved function_call is treated as not-applicable."""
        self.run_intermediate_response_not_applicable_test()


@pytest.mark.unittest
class TestToolSelectionValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    ToolDefinitionExtractionUnitTests,
    LogSafeSummaryUnitTests,
    ConversationHistoryReformatUnitTests,
):
    """Low-level unit tests for tool_selection's repeated validators, utils and methods."""

    evaluator_class = ToolSelectionEvaluator


@pytest.mark.unittest
class TestToolSelectionInternalBranches:
    """Cover tool_selection helper, conversion and eval branches not hit elsewhere."""

    _TOOL_CALL_RESPONSE = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "tool_call_id": "c1", "name": "search", "arguments": {"q": "x"}}
            ],
        }
    ]
    _TEXT_RESPONSE = [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]

    def test_validate_tool_calls_rejects_non_dict_item(self):
        """Return an error when a tool call is not a dictionary."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        assert isinstance(evaluator._validator._validate_tool_calls([123]), EvaluationException)

    def test_extract_tool_names_handles_function_and_direct_shapes(self):
        """Extract names from both the function-call and direct-name tool call shapes."""
        names = _extract_tool_names_from_calls([{"function": {"name": "f1"}}, {"name": "f2"}])
        assert names == ["f1", "f2"]

    def test_get_built_in_tool_definition_unknown_returns_none(self):
        """Return None for an unknown built-in tool name."""
        assert _get_built_in_tool_definition("nonexistent_tool") is None

    def test_format_value_returns_non_string_unchanged(self):
        """Return non-string, non-None values unchanged."""
        assert _format_value(123) == 123

    def test_get_conversation_history_skips_role_less_message(self):
        """Skip role-less messages and validate turn balance before returning."""
        query = [
            {"content": "no role"},
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
        with pytest.raises(EvaluationException):
            _get_conversation_history(query)

    def test_log_safe_summary_handles_raising_object(self):
        """Return a safe placeholder when summarizing an object whose length raises."""
        class _Bad:
            def __len__(self):
                raise RuntimeError("boom")

        assert "summary unavailable" in _log_safe_summary(_Bad())

    def test_convert_kwargs_no_tool_calls_in_response(self):
        """Return the no-tool-calls error when the response has no tool calls."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = evaluator._convert_kwargs_to_eval_input(query="q", response=self._TEXT_RESPONSE)
        assert result["error_message"] == evaluator._NO_TOOL_CALLS_MESSAGE

    def test_convert_kwargs_wraps_dict_definitions(self):
        """Wrap dict-shaped tool definitions when the response is a string."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response="just a string response",
            tool_definitions={"name": "x", "type": "function", "parameters": {}},
        )
        assert isinstance(result, dict)

    def test_convert_kwargs_extraction_fails_no_definitions(self):
        """Return the no-definitions error when extraction fails with an empty list."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response=[dict(m) for m in self._TOOL_CALL_RESPONSE], tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_convert_kwargs_extraction_fails_unmatched_definitions(self):
        """Return the missing-definitions error when a used tool has no definition."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q",
            response=[dict(m) for m in self._TOOL_CALL_RESPONSE],
            tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
        )
        assert result["error_message"] == evaluator._TOOL_DEFINITIONS_MISSING_MESSAGE

    def test_convert_kwargs_string_response_empty_definitions(self):
        """Return the no-definitions error when a string response has no definitions."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = evaluator._convert_kwargs_to_eval_input(
            query="q", response="just a string response", tool_definitions=[]
        )
        assert result["error_message"] == evaluator._NO_TOOL_DEFINITIONS_MESSAGE

    def test_do_eval_missing_query_raises(self):
        """Raise when ``_do_eval`` is invoked without a query."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({}))

    def test_do_eval_invalid_output_raises(self):
        """Raise when the judge returns a non-dict ``llm_output`` payload."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")

        async def _bad_flow(**kwargs):
            return {"llm_output": "not-a-dict"}

        evaluator._flow = _bad_flow
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "q"}))

    def test_real_call_error_message_returns_not_applicable(self):
        """Return a not-applicable result when input conversion yields an error message."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        result = asyncio.run(
            evaluator._real_call(
                query="q",
                response=[dict(m) for m in self._TEXT_RESPONSE],
                tool_definitions=[{"name": "other", "type": "function", "parameters": {}}],
            )
        )
        assert result["tool_selection"] is None

    def test_calculate_tool_selection_accuracy_with_calls(self):
        """Compute selection accuracy as a percentage when tools were called."""
        evaluator = create_mocked_evaluator(ToolSelectionEvaluator, "tool_selection")
        accuracy = evaluator._calculate_tool_selection_accuracy(
            {"correct_tool_selections": 3, "wrong_tool_selections": 1}
        )
        assert accuracy == 75.0
