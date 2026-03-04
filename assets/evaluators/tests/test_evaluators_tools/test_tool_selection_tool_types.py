# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Tool Selection Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from typing import List
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ..test_evaluators_tools import tool_type_test_data as data
from ...builtin.tool_selection.evaluator._tool_selection import (
    ToolSelectionEvaluator,
)


@pytest.mark.unittest
class TestToolSelectionToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for ToolSelectionEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.
    """

    #region Expected flow inputs for each test
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
    #endregion

    evaluator_type = ToolSelectionEvaluator

    is_tool_definition_required = True

    _additional_expected_field_suffixes = ["details"]

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tool type evaluator tests."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ] + [f"{self._result_prefix}_{suffix}" for suffix in self._additional_expected_field_suffixes]
