# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import pytest
from typing import List
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolCallAccuracyEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Tool Call Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_QUERY,
        "tool_calls": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.LOCAL_CALLS_TOOL_DEFINITIONS,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_QUERY,
        "tool_calls": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.FILE_SEARCH_TOOL_DEFINITIONS,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_QUERY,
        "tool_calls": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.IMAGE_GEN_TOOL_DEFINITIONS,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_QUERY,
        "tool_calls": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MEMORY_SEARCH_TOOL_DEFINITIONS,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_QUERY,
        "tool_calls": data.KB_MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.KB_MCP_TOOL_DEFINITIONS,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_QUERY,
        "tool_calls": data.MCP_EXPECTED_FLOW_TOOL_CALLS,
        "tool_definitions": data.MCP_TOOL_DEFINITIONS,
    }
    # endregion

    evaluator_type = ToolCallAccuracyEvaluator

    check_for_unsupported_tools = True

    is_tool_definition_required = True

    MINIMAL_RESPONSE = BaseToolCallEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tools evaluators."""
        return [
            f"{self.result_key}",
            f"{self.result_key}_score",
            f"{self.result_key}_result",
            f"{self.result_key}_passed",
            f"{self.result_key}_reason",
            f"{self.result_key}_status",
            f"{self.result_key}_threshold",
            f"{self.result_key}_properties",
        ]
