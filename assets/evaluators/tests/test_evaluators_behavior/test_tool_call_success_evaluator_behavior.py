# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Call Success Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.tool_call_success.evaluator._tool_call_success import (
    ToolCallSuccessEvaluator,
)


@pytest.mark.unittest
class TestToolCallSuccessEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Call Success Evaluator.

    Tests different input formats and scenarios.
    """

    #region Expected flow inputs for each test
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
    #endregion

    evaluator_type = ToolCallSuccessEvaluator

    check_for_unsupported_tools = True

    # Test Configs
    requires_query = False

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_results_without_arguments
