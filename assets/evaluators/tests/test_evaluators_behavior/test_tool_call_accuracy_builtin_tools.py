# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator with Azure built-in tools.

Tests that the evaluator correctly processes conversations containing
built-in tool calls (code_interpreter, bing_grounding, file_search, etc.)
and auto-injects the correct tool definitions into the LLM flow input.
"""

import pytest
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)
from .tool_type_test_data import BuiltInToolData, FunctionToolData


@pytest.mark.unittest
class TestToolCallAccuracyBuiltinTools(BaseToolTypeEvaluatorTest):
    """
    Tests for ToolCallAccuracyEvaluator with Azure built-in tools.

    Verifies that:
    1. The evaluator succeeds with built-in tool conversations
    2. Built-in tool definitions are auto-injected correctly
    3. The correct input is sent to _flow
    """

    evaluator_type = ToolCallAccuracyEvaluator

    # ==================== CODE INTERPRETER ====================

    def test_code_interpreter_with_tool_calls(self):
        """Code interpreter tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - code_interpreter via tool_calls",
            query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            tool_calls=BuiltInToolData.TOOL_CALLS_CODE_INTERPRETER,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_CODE_INTERPRETER,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_CODE_INTERPRETER,
            ],
        )

    def test_code_interpreter_with_response(self):
        """Code interpreter tool call extracted from response."""
        self.run_tool_type_test(
            test_label="Built-in - code_interpreter via response",
            query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            response=BuiltInToolData.RESPONSE_CODE_INTERPRETER,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_CODE_INTERPRETER,
            ],
        )

    # ==================== BING GROUNDING ====================

    def test_bing_grounding_with_tool_calls(self):
        """Bing grounding tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - bing_grounding via tool_calls",
            query=BuiltInToolData.QUERY_BING_GROUNDING,
            tool_calls=BuiltInToolData.TOOL_CALLS_BING_GROUNDING,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_BING_GROUNDING,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_BING_GROUNDING,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_BING_GROUNDING,
            ],
        )

    def test_bing_grounding_with_response(self):
        """Bing grounding tool call extracted from response."""
        self.run_tool_type_test(
            test_label="Built-in - bing_grounding via response",
            query=BuiltInToolData.QUERY_BING_GROUNDING,
            response=BuiltInToolData.RESPONSE_BING_GROUNDING,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_BING_GROUNDING,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_BING_GROUNDING,
            ],
        )

    # ==================== BING CUSTOM SEARCH ====================

    def test_bing_custom_search_with_tool_calls(self):
        """Bing custom search tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - bing_custom_search via tool_calls",
            query=BuiltInToolData.QUERY_BING_CUSTOM_SEARCH,
            tool_calls=BuiltInToolData.TOOL_CALLS_BING_CUSTOM_SEARCH,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_BING_CUSTOM_SEARCH,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_BING_CUSTOM_SEARCH,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_BING_CUSTOM_SEARCH,
            ],
        )

    # ==================== FILE SEARCH ====================

    def test_file_search_with_tool_calls(self):
        """File search tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - file_search via tool_calls",
            query=BuiltInToolData.QUERY_FILE_SEARCH,
            tool_calls=BuiltInToolData.TOOL_CALLS_FILE_SEARCH,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_FILE_SEARCH,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_FILE_SEARCH,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_FILE_SEARCH,
            ],
        )

    def test_file_search_with_response(self):
        """File search tool call extracted from response."""
        self.run_tool_type_test(
            test_label="Built-in - file_search via response",
            query=BuiltInToolData.QUERY_FILE_SEARCH,
            response=BuiltInToolData.RESPONSE_FILE_SEARCH,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_FILE_SEARCH,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_FILE_SEARCH,
            ],
        )

    # ==================== AZURE AI SEARCH ====================

    def test_azure_ai_search_with_tool_calls(self):
        """Azure AI Search tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - azure_ai_search via tool_calls",
            query=BuiltInToolData.QUERY_AZURE_AI_SEARCH,
            tool_calls=BuiltInToolData.TOOL_CALLS_AZURE_AI_SEARCH,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_AZURE_AI_SEARCH,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_AZURE_AI_SEARCH,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_AZURE_AI_SEARCH,
            ],
        )

    # ==================== SHAREPOINT GROUNDING ====================

    def test_sharepoint_grounding_with_tool_calls(self):
        """SharePoint grounding tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - sharepoint_grounding via tool_calls",
            query=BuiltInToolData.QUERY_SHAREPOINT_GROUNDING,
            tool_calls=BuiltInToolData.TOOL_CALLS_SHAREPOINT_GROUNDING,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_SHAREPOINT_GROUNDING,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_SHAREPOINT_GROUNDING,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_SHAREPOINT_GROUNDING,
            ],
        )

    # ==================== FABRIC DATA AGENT ====================

    def test_fabric_dataagent_with_tool_calls(self):
        """Fabric data agent tool call with auto-injected definition."""
        self.run_tool_type_test(
            test_label="Built-in - fabric_dataagent via tool_calls",
            query=BuiltInToolData.QUERY_FABRIC_DATAAGENT,
            tool_calls=BuiltInToolData.TOOL_CALLS_FABRIC_DATAAGENT,
            tool_definitions=[],
            expected_flow_query=BuiltInToolData.QUERY_FABRIC_DATAAGENT,
            expected_flow_tool_calls=BuiltInToolData.TOOL_CALLS_FABRIC_DATAAGENT,
            expected_flow_tool_definition_contains=[
                BuiltInToolData.EXPECTED_DEFINITION_FABRIC_DATAAGENT,
            ],
        )

    # ==================== MIXED: BUILT-IN + FUNCTION TOOL ====================

    def test_mixed_builtin_and_function_tool(self):
        """Mix of built-in tool (code_interpreter) and function tool (get_horoscope).

        Verifies that both auto-injected and user-provided definitions appear in _flow input.
        """
        mixed_tool_calls = (
            BuiltInToolData.TOOL_CALLS_CODE_INTERPRETER + FunctionToolData.TOOL_CALLS_HOROSCOPE
        )
        self.run_tool_type_test(
            test_label="Mixed - built-in + function tool",
            query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            tool_calls=mixed_tool_calls,
            tool_definitions=FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE,
            expected_flow_query=BuiltInToolData.QUERY_CODE_INTERPRETER,
            expected_flow_tool_calls=mixed_tool_calls,
            # Should contain both the user-provided function def and auto-injected built-in def
            expected_flow_tool_definition_contains=[
                FunctionToolData.TOOL_DEFINITIONS_HOROSCOPE[0],
                BuiltInToolData.EXPECTED_DEFINITION_CODE_INTERPRETER,
            ],
            expected_flow_tool_definitions_count=2,
        )
