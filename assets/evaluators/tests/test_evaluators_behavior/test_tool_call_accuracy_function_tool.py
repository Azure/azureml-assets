# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator with function tools.

Tests that the evaluator correctly processes conversations containing
user-defined function tool calls (e.g., get_horoscope) and sends the
correct input to the LLM flow.
"""

import pytest
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)
from .tool_type_test_data import FunctionToolData


@pytest.mark.unittest
class TestToolCallAccuracyFunctionTool(BaseToolTypeEvaluatorTest):
    """
    Tests for ToolCallAccuracyEvaluator with user-defined function tools.

    Verifies that:
    1. The evaluator succeeds with function tool conversations
    2. The correct tool definitions, tool calls, and query are sent to _flow
    """

    evaluator_type = ToolCallAccuracyEvaluator

    # ==================== SINGLE FUNCTION CALL TESTS ====================

    def test_single_function_call_with_tool_calls_param(self):
        """Single function call provided via tool_calls parameter."""
        self.run_tool_type_test(
            test_label="Function Tool - Single call via tool_calls param",
            query=FunctionToolData.QUERY_HOROSCOPE,
            tool_calls=FunctionToolData.TOOL_CALLS_HOROSCOPE,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
            expected_flow_query=FunctionToolData.QUERY_HOROSCOPE,
            expected_flow_tool_calls=FunctionToolData.TOOL_CALLS_HOROSCOPE,
            expected_flow_tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
        )

    def test_single_function_call_with_response(self):
        """Single function call extracted from response conversation."""
        self.run_tool_type_test(
            test_label="Function Tool - Single call via response",
            query=FunctionToolData.QUERY_HOROSCOPE,
            response=FunctionToolData.RESPONSE_HOROSCOPE,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
            # When response is provided, tool calls are parsed from it
            expected_flow_query=FunctionToolData.QUERY_HOROSCOPE,
            expected_flow_tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
        )

    def test_single_function_call_with_both_response_and_tool_calls(self):
        """Single function call with both response and tool_calls provided.

        When both are provided, tool calls from response take precedence.
        """
        self.run_tool_type_test(
            test_label="Function Tool - Single call with response + tool_calls",
            query=FunctionToolData.QUERY_HOROSCOPE,
            response=FunctionToolData.RESPONSE_HOROSCOPE,
            tool_calls=FunctionToolData.TOOL_CALLS_HOROSCOPE,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
            expected_flow_query=FunctionToolData.QUERY_HOROSCOPE,
            expected_flow_tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
        )

    # ==================== MULTIPLE FUNCTION CALLS TESTS ====================

    def test_multiple_function_calls_with_tool_calls_param(self):
        """Multiple function calls provided via tool_calls parameter."""
        self.run_tool_type_test(
            test_label="Function Tool - Multiple calls via tool_calls param",
            query=FunctionToolData.QUERY_HOROSCOPE,
            tool_calls=FunctionToolData.TOOL_CALLS_MULTI,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_MULTI,
            expected_flow_query=FunctionToolData.QUERY_HOROSCOPE,
            expected_flow_tool_calls=FunctionToolData.TOOL_CALLS_MULTI,
            expected_flow_tool_definitions=FunctionToolData.TOOL_DEFINITIONS_MULTI,
        )

    def test_multiple_function_calls_with_response(self):
        """Multiple function calls extracted from response conversation."""
        self.run_tool_type_test(
            test_label="Function Tool - Multiple calls via response",
            query=FunctionToolData.QUERY_HOROSCOPE,
            response=FunctionToolData.RESPONSE_MULTI,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_MULTI,
            expected_flow_query=FunctionToolData.QUERY_HOROSCOPE,
            expected_flow_tool_definitions=FunctionToolData.TOOL_DEFINITIONS_MULTI,
        )

    # ==================== FLOW INPUT FORMAT VERIFICATION ====================

    def test_flow_receives_correct_tool_definition_count(self):
        """Verify _flow receives exactly the expected number of tool definitions."""
        self.run_tool_type_test(
            test_label="Function Tool - Tool definition count",
            query=FunctionToolData.QUERY_HOROSCOPE,
            tool_calls=FunctionToolData.TOOL_CALLS_HOROSCOPE,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
            expected_flow_tool_definitions_count=1,
        )

    def test_flow_receives_correct_multi_tool_definition_count(self):
        """Verify _flow receives correct count for multiple tool definitions."""
        self.run_tool_type_test(
            test_label="Function Tool - Multi tool definition count",
            query=FunctionToolData.QUERY_HOROSCOPE,
            tool_calls=FunctionToolData.TOOL_CALLS_MULTI,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_MULTI,
            expected_flow_tool_definitions_count=2,
        )
