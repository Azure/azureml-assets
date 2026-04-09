# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Selection Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import pytest
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.tool_selection.evaluator._tool_selection import (
    ToolSelectionEvaluator,
)


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

    # OpenAPI: ToolSelection flow is not called (no extractable tool calls)
    # Expected flow inputs not used since the test will not reach flow assertion

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

    def test_openapi(self):
        """OpenAPI: ToolSelection flow is not called (no extractable tool calls)."""
        results, flow_mock = self._run_evaluation_and_return_mocked_flow(
            query=data.OPENAPI_QUERY,
            response=data.OPENAPI_RESPONSE,
            tool_definitions=data.OPENAPI_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "OpenAPI")
        self.assert_not_applicable(result_data)
        assert flow_mock is not None, "Flow mock should be set when use_mocking=True"
        flow_mock.assert_not_called()
