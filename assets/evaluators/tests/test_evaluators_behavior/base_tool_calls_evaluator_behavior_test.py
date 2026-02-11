# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of tool calls evaluators.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import json
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest


class BaseToolCallEvaluatorBehaviorTest(BaseToolsEvaluatorBehaviorTest):
    """
    Base class for tool call evaluator behavioral tests with tool_calls.

    Extends BaseToolsEvaluatorBehaviorTest with tool call support.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "ToolSelection")
    Subclasses may override:
    - requires_tool_definitions: bool - whether tool definitions are required
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    """

    _additional_expected_field_suffixes = ["details"]

    # Test Configs
    requires_tool_definitions = True

    # region Test Data
    # Minimal valid response format for tool call evaluators
    MINIMAL_RESPONSE = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "fetch_weather",
                    "arguments": {"location": "Seattle"},
                    "tool_call_id": "call_1",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "send_email",
                    "arguments": {},
                    "tool_call_id": "call_2",
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

    INVALID_TOOL_CALLS_AS_STRING: str = json.dumps(INVALID_TOOL_CALLS)
    # endregion

    # ==================== TOOL CALLS WITHOUT RESPONSE TESTS ====================

    def run_tool_calls_test(
        self, input_tool_calls, description: str, assert_type: BaseToolsEvaluatorBehaviorTest.AssertType
    ):
        """Test various tool calls inputs."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=None,
            tool_calls=input_tool_calls,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, description)

        self.assert_expected_behavior(assert_type, result_data)

    def test_tool_calls_invalid_format_as_string(self):
        """Tool calls as string - should pass."""
        self.run_tool_calls_test(
            input_tool_calls=self.INVALID_TOOL_CALLS_AS_STRING,
            description="Tool Calls Invalid Format as String",
            assert_type=self.AssertType.PASS,
        )

    def test_tool_calls_invalid_format(self):
        """Tool calls in invalid format - should raise invalid value error."""
        self.run_tool_calls_test(
            input_tool_calls=self.INVALID_TOOL_CALLS,
            description="Tool Calls Invalid Format",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_wrong_type(self):
        """Tool calls in wrong type - should raise invalid value error."""
        self.run_tool_calls_test(
            input_tool_calls=self.WRONG_TYPE,
            description="Tool Calls Wrong Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_empty_list(self):
        """Tool calls as empty list - should raise missing field error."""
        self.run_tool_calls_test(
            input_tool_calls=self.EMPTY_LIST,
            description="Tool Calls Empty List",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    # ==================== RESPONSE WITH VALID TOOL CALLS TESTS ====================

    def run_response_with_valid_tool_calls_test(self, input_response, description: str):
        """Test various response inputs with valid tool calls."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=input_response,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, description)

        self.assert_expected_behavior(self.AssertType.PASS, result_data)

    def test_response_not_present_with_valid_tool_calls(self):
        """Response not present with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=None, description="Response Not Present with Valid Tool Calls"
        )

    def test_response_as_string_with_valid_tool_calls(self):
        """Response as string with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=self.STRING_RESPONSE, description="Response String with Valid Tool Calls"
        )

    def test_response_invalid_format_as_string_with_valid_tool_calls(self):
        """Response as string with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=self.INVALID_RESPONSE_AS_STRING,
            description="Response Invalid Format as String with Valid Tool Calls",
        )

    def test_response_invalid_format_with_valid_tool_calls(self):
        """Response in invalid format with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=self.INVALID_RESPONSE, description="Response Invalid Format with Valid Tool Calls"
        )

    def test_response_wrong_type_with_valid_tool_calls(self):
        """Response in wrong type with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=self.WRONG_TYPE, description="Response Wrong Type with Valid Tool Calls"
        )

    def test_response_empty_list_with_valid_tool_calls(self):
        """Response as empty list with valid tool calls - should pass."""
        self.run_response_with_valid_tool_calls_test(
            input_response=self.EMPTY_LIST, description="Response Empty List with Valid Tool Calls"
        )

    # ==================== TOOL CALLS WITH VALID RESPONSE TESTS ====================

    def run_tool_calls_with_valid_response_test(self, input_tool_calls, description: str):
        """Test various tool call inputs with valid response."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=input_tool_calls,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, description)

        self.assert_expected_behavior(self.AssertType.PASS, result_data)

    def test_tool_calls_not_present_with_valid_response(self):
        """Tool calls not present with valid response - should pass."""
        self.run_tool_calls_with_valid_response_test(
            input_tool_calls=None, description="Tool Calls Not Present with Valid Response"
        )

    def test_tool_calls_invalid_format_as_string_with_valid_response(self):
        """Tool calls as string with valid response - should pass."""
        self.run_tool_calls_with_valid_response_test(
            input_tool_calls=self.INVALID_TOOL_CALLS_AS_STRING,
            description="Tool Calls Invalid Format as String with Valid Response",
        )

    def test_tool_calls_invalid_format_with_valid_response(self):
        """Tool calls in invalid format with valid response - should pass."""
        self.run_tool_calls_with_valid_response_test(
            input_tool_calls=self.INVALID_TOOL_CALLS, description="Tool Calls Invalid Format with Valid Response"
        )

    def test_tool_calls_wrong_type_with_valid_response(self):
        """Tool calls in wrong type with valid response - should pass."""
        self.run_tool_calls_with_valid_response_test(
            input_tool_calls=self.WRONG_TYPE, description="Tool Calls Wrong Type with Valid Response"
        )

    def test_tool_calls_empty_list_with_valid_response(self):
        """Tool calls as empty list with valid response - should pass."""
        self.run_tool_calls_with_valid_response_test(
            input_tool_calls=self.EMPTY_LIST, description="Tool Calls Empty List with Valid Response"
        )

    # ==================== TOOL CALLS PARAMETERS TESTS ====================

    def test_tool_definitions_missing_type_parameter(self):
        """Tool calls missing 'type' parameter - should raise invalid value error."""
        modified_tool_calls = self.remove_parameter_from_input(input_data=self.VALID_TOOL_CALLS, parameter_name="type")
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Missing 'type'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_invalid_type_parameter(self):
        """Tool calls with invalid 'type' parameter - should raise invalid value error."""
        modified_tool_calls = self.update_parameter_in_input(
            input_data=self.VALID_TOOL_CALLS, parameter_name="type", parameter_value="invalid_type"
        )
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Invalid 'type'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_missing_tool_call_id_parameter(self):
        """Tool calls missing 'tool_call_id' parameter - should raise invalid value error."""
        modified_tool_calls = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_CALLS, parameter_name="tool_call_id"
        )
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Missing 'tool_call_id'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_missing_name_parameter(self):
        """Tool calls missing 'name' parameter - should raise invalid value error."""
        modified_tool_calls = self.remove_parameter_from_input(input_data=self.VALID_TOOL_CALLS, parameter_name="name")
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Missing 'name'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_missing_arguments_parameter(self):
        """Tool calls missing 'arguments' parameter - should raise invalid value error."""
        modified_tool_calls = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_CALLS, parameter_name="arguments"
        )
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Missing 'arguments'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_calls_invalid_arguments_type(self):
        """Tool calls with invalid 'arguments' type - should raise invalid value error."""
        modified_tool_calls = self.update_parameter_in_input(
            input_data=self.VALID_TOOL_CALLS, parameter_name="arguments", parameter_value=self.WRONG_TYPE
        )
        self.run_tool_calls_test(
            input_tool_calls=modified_tool_calls,
            description="Tool Calls Invalid 'arguments' Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )
