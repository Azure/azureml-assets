# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of tool calls evaluators.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

from .base_tool_parameters_behavior_test import BaseToolParametersBehaviorTest


class BaseToolCallEvaluatorBehaviorTest(BaseToolParametersBehaviorTest):
    """
    Base class for tool call evaluator behavioral tests with tool_calls.

    Extends BaseToolParametersBehaviorTest with tool call support.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "ToolSelection")
    Subclasses may override:
    - requires_valid_format: bool - whether valid format is required for response
    - requires_tool_definitions: bool - whether tool definitions are required
    - requires_arguments: bool - whether tool calls need arguments to be valid
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    """

    # Test Configs
    requires_valid_format: bool = True
    requires_tool_definitions: bool = True

    MINIMAL_RESPONSE = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "fetch_weather",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "send_email",
                }
            ],
        },
    ]

    # Tool call test data
    VALID_TOOL_CALLS = [
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "name": "fetch_weather",
            "arguments": {"location": "Seattle"},
        },
        {
            "type": "tool_call",
            "tool_call_id": "call_2",
            "name": "send_email",
            "arguments": {
                "recipient": "your_email@example.com",
                "subject": "Weather Information for Seattle",
                "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
            },
        },
    ]

    INVALID_TOOL_CALLS = [
        {
            "name": "fetch_weather",
            "arguments": {"location": "Seattle"},
        },
        {
            "name": "send_email",
            "arguments": {
                "recipient": "your_email@example.com",
                "subject": "Weather Information for Seattle",
                "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
            },
        },
    ]

    # ==================== RESPONSE AND TOOL CALLS TESTS ====================

    def test_response_not_present_with_tool_calls(self):
        """Response not present but tool_calls provided - should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=None,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Response Not Present with Tool Calls")

        self.assert_pass(result_data)

    def test_response_as_string_with_tool_calls(self):
        """Response as string - should pass with explicit tool_calls."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.STRING_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Response String with Tool Calls")

        self.assert_pass(result_data)

    def test_response_invalid_format_with_tool_calls(self):
        """Response in invalid format - should pass with explicit tool_calls."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.INVALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Response Invalid With Tool Calls")

        self.assert_pass(result_data)

    # ==================== TOOL CALLS TESTS ====================

    def test_tool_calls_not_present(self):
        """Tool calls not present with valid response - should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=None,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Tool Calls Not Present")

        self.assert_pass(result_data)

    def test_tool_calls_invalid_format(self):
        """Tool calls in invalid format with valid response - should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.INVALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Tool Calls Invalid With Valid Response")

        self.assert_pass(result_data)

    def test_tool_calls_invalid_format_without_valid_response(self):
        """Tool calls in invalid format without valid response - should return not_applicable."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=None,
            tool_calls=self.INVALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Tool Calls Invalid Without Valid Response")

        if self.requires_valid_format:
            self.assert_error(result_data)
        else:
            self.assert_pass(result_data)
