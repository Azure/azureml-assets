# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of evaluators.

Tests various input scenarios: query, and response.
"""

from typing import Any, Dict, List
import json
from .base_evaluator_runner import BaseEvaluatorRunner


class BaseEvaluatorBehaviorTest(BaseEvaluatorRunner):
    """
    Base class for evaluator behavioral tests with query and response inputs.

    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "Relevance")
    Subclasses may override:
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    """

    # Subclasses may override
    # Test Configs
    requires_query: bool = True

    # region Common Test Data
    # Tool-related test data - can be overridden by subclasses
    VALID_TOOL_CALLS: List[Dict[str, Any]] = None
    INVALID_TOOL_CALLS: List[Dict[str, Any]] = None
    VALID_TOOL_DEFINITIONS: List[Dict[str, Any]] = None
    INVALID_TOOL_DEFINITIONS: List[Dict[str, Any]] = None

    # Minimal valid response format for the evaluator - can be overridden by subclasses
    MINIMAL_RESPONSE: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I have successfully sent you an email with the weather information for Seattle. \
                        The current weather is rainy with a temperature of 14\u00b0C.",
                }
            ],
        },
    ]

    # Common test data
    weather_tool_call_and_assistant_response: List[Dict[str, Any]] = [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14\u00b0C"}}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I have successfully sent you an email with the weather information for Seattle. \
                        The current weather is rainy with a temperature of 14\u00b0C.",
                }
            ],
        },
    ]

    email_tool_call_and_assistant_response: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "send_email",
                    "arguments": {
                        "recipient": "your_email@example.com",
                        "subject": "Weather Information for Seattle",
                        "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
                    },
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I have successfully sent you an email with the weather information for Seattle. \
                        The current weather is rainy with a temperature of 14\u00b0C.",
                }
            ],
        },
    ]

    tool_results_without_arguments: List[Dict[str, Any]] = [
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14\u00b0C"}}],
        },
        {
            "tool_call_id": "call_2",
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"message": "Email successfully sent to your_email@example.com."},
                }
            ],
        },
    ]

    tool_results_with_arguments: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "fetch_weather",
                    "arguments": {"location": "Seattle"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "name": "send_email",
                    "arguments": {
                        "recipient": "your_email@example.com",
                        "subject": "Weather Information for Seattle",
                        "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
                    },
                }
            ],
        },
    ]

    # Wrong type input
    WRONG_TYPE = 1

    # Empty list input
    EMPTY_LIST: List[Any] = []

    # Common test data for query and response
    STRING_QUERY: str = "Can you send me an email to your_email@example.com with weather information for Seattle?"

    STRING_RESPONSE: str = (
        "I have successfully sent you an email with the weather information for Seattle. "
        "The current weather is rainy with a temperature of 14\u00b0C."
    )

    VALID_QUERY: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": {
                "type": "input_text",
                "text": "Can you send me an email to your_email@example.com with weather information for Seattle?",
            },
        },
    ]

    VALID_RESPONSE: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "name": "fetch_weather",
                    "arguments": {"location": "Seattle"},
                }
            ],
        },
        {
            "tool_call_id": "call_1",
            "role": "tool",
            "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14\u00b0C"}}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_2",
                    "name": "send_email",
                    "arguments": {
                        "recipient": "your_email@example.com",
                        "subject": "Weather Information for Seattle",
                        "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
                    },
                }
            ],
        },
        {
            "tool_call_id": "call_2",
            "role": "tool",
            "content": [
                {
                    "type": "tool_result",
                    "tool_result": {"message": "Email successfully sent to your_email@example.com."},
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "I have successfully sent you an email with the weather information for Seattle. \
                        The current weather is rainy with a temperature of 14\u00b0C.",
                }
            ],
        },
    ]

    INVALID_QUERY: List[Dict[str, Any]] = [
        {
            "user": "Can you send me an email to your_email@example.com with weather information for Seattle?",
        },
    ]

    INVALID_RESPONSE: List[Dict[str, Any]] = [
        {
            "tool_call": [
                {
                    "name": "fetch_weather",
                    "arguments": {"location": "Seattle"},
                }
            ],
        },
        {
            "tool_result": {"weather": "Rainy, 14\u00b0C"},
        },
        {
            "tool_call": [
                {
                    "name": "send_email",
                    "arguments": {
                        "recipient": "your_email@example.com",
                        "subject": "Weather Information for Seattle",
                        "body": "The current weather in Seattle is rainy with a temperature of 14\u00b0C.",
                    },
                }
            ],
        },
        {
            "tool_result": {"message": "Email successfully sent to your_email@example.com."},
        },
        {
            "assistant": "I have successfully sent you an email with the weather information for Seattle. \
                The current weather is rainy with a temperature of 14\u00b0C.",
        },
    ]

    INVALID_QUERY_AS_STRING: str = json.dumps(INVALID_QUERY)

    INVALID_RESPONSE_AS_STRING: str = json.dumps(INVALID_RESPONSE)
    # endregion

    def remove_parameter_from_input_content(self, input_data: List[Dict], parameter_name: str) -> List[Dict]:
        """Remove a parameter from the content field of all items in the input data."""
        input_data_copy = input_data.copy()

        for message in input_data_copy:
            for content in message.get("content", []):
                if parameter_name in content:
                    del content[parameter_name]

        return input_data_copy

    def remove_parameter_from_input(self, input_data: List[Dict], parameter_name: str) -> List[Dict]:
        """Remove a parameter from all items in the input data."""
        input_data_copy = input_data.copy()

        for message in input_data_copy:
            if parameter_name in message:
                del message[parameter_name]

        return input_data_copy

    def update_parameter_in_input_content(
        self, input_data: List[Dict], parameter_name: str, parameter_value: Any
    ) -> List[Dict]:
        """Update parameter value for the content field of all items in the input data."""
        input_data_copy = input_data.copy()

        for message in input_data_copy:
            for content in message.get("content", []):
                if parameter_name in content:
                    content[parameter_name] = parameter_value

        return input_data_copy

    def update_parameter_in_input(
        self, input_data: List[Dict], parameter_name: str, parameter_value: Any
    ) -> List[Dict]:
        """Update parameter value for all items in the input data."""
        input_data_copy = input_data.copy()

        for message in input_data_copy:
            if parameter_name in message:
                message[parameter_name] = parameter_value

        return input_data_copy

    # ==================== All Valid ====================

    def test_all_valid_inputs(self):
        """All inputs valid and in correct format."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "All Valid")

        self.assert_pass(result_data)

    # ==================== QUERY TESTS ====================

    def test_query(self, input_query, description: str, assert_type: BaseEvaluatorRunner.AssertType):
        """Helper method to test various query inputs."""
        results = self._run_evaluation(
            query=input_query,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, description)

        if self.requires_query:
            self.assert_expected_behavior(assert_type, result_data)
        else:
            self.assert_pass(result_data)

    def test_query_not_present(self):
        """Query not present - should raise missing field error."""
        self.test_query(input_query=None, description="Query Not Present", assert_type=self.AssertType.MISSING_FIELD)

    def test_query_as_string(self):
        """Query as string - should pass."""
        self.test_query(input_query=self.STRING_QUERY, description="Query String", assert_type=self.AssertType.PASS)

    def test_query_invalid_format_as_string(self):
        """Query as string - should pass."""
        self.test_query(
            input_query=self.INVALID_QUERY_AS_STRING,
            description="Query Invalid Format as String",
            assert_type=self.AssertType.PASS,
        )

    def test_query_invalid_format(self):
        """Query in invalid format - should raise invalid value error."""
        self.test_query(
            input_query=self.INVALID_QUERY,
            description="Query Invalid Format",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_wrong_type(self):
        """Query in wrong type - should raise invalid value error."""
        self.test_query(
            input_query=self.WRONG_TYPE, description="Query Wrong Type", assert_type=self.AssertType.INVALID_VALUE
        )

    def test_query_empty_list(self):
        """Query as empty list - should raise missing field error."""
        self.test_query(
            input_query=self.EMPTY_LIST, description="Query Empty List", assert_type=self.AssertType.MISSING_FIELD
        )

    # ==================== QUERY PARAMETER TESTS ====================

    def test_query_missing_role_parameter(self):
        """Query is missing role parameter - should raise invalid value error."""
        modified_query = self.remove_parameter_from_input_content(self.VALID_QUERY, "role")
        self.test_query(
            input_query=modified_query,
            description="Query Missing Role Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_valid_system_role_parameter(self):
        """Query with system role parameter - should pass."""
        modified_query = self.update_parameter_in_input(self.VALID_QUERY, "role", "system")
        self.test_query(
            input_query=modified_query,
            description="Query Valid System Role Parameter",
            assert_type=self.AssertType.PASS,
        )

    def test_query_invalid_role_parameter(self):
        """Query with invalid role parameter - should raise invalid value error."""
        modified_query = self.update_parameter_in_input(self.VALID_QUERY, "role", "invalid_role")
        # TODO: Should this be INVALID_VALUE instead of PASS?
        self.test_query(
            input_query=modified_query, description="Query Invalid Role Parameter", assert_type=self.AssertType.PASS
        )

    def test_query_missing_content_parameter(self):
        """Query is missing content parameter - should raise invalid value error."""
        modified_query = self.remove_parameter_from_input(self.VALID_QUERY, "content")
        self.test_query(
            input_query=modified_query,
            description="Query Missing Content Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_empty_content_parameter(self):
        """Query has empty content parameter - should raise invalid value error."""
        modified_query = self.update_parameter_in_input(self.VALID_QUERY, "content", self.EMPTY_LIST)
        self.test_query(
            input_query=modified_query,
            description="Query Empty Content Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_invalid_content_parameter_type(self):
        """Query has invalid content parameter type - should raise invalid value error."""
        modified_query = self.update_parameter_in_input(self.VALID_QUERY, "content", self.WRONG_TYPE)
        self.test_query(
            input_query=modified_query,
            description="Query Invalid Content Parameter Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_valid_content_parameter(self):
        """Query has valid content parameter - should pass."""
        modified_query = self.update_parameter_in_input(self.VALID_QUERY, "content", self.STRING_QUERY)
        self.test_query(
            input_query=modified_query, description="Query Valid Content Parameter", assert_type=self.AssertType.PASS
        )

    def test_query_missing_text_parameter_in_content(self):
        """Query is missing text parameter in content - should raise invalid value error."""
        modified_query = self.remove_parameter_from_input_content(self.VALID_QUERY, "text")
        self.test_query(
            input_query=modified_query,
            description="Query Missing Text Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_missing_type_parameter_in_content(self):
        """Query is missing type parameter in content - should raise invalid value error."""
        modified_query = self.remove_parameter_from_input_content(self.VALID_QUERY, "type")
        self.test_query(
            input_query=modified_query,
            description="Query Missing Type Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_invalid_type_parameter_in_content(self):
        """Query has invalid type parameter in content - should raise invalid value error."""
        modified_query = self.update_parameter_in_input_content(self.VALID_QUERY, "type", "invalid_type")
        self.test_query(
            input_query=modified_query,
            description="Query Invalid Type Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_query_valid_type_parameter_in_content(self):
        """Query has valid type parameter in content - should pass."""
        modified_query = self.update_parameter_in_input_content(self.VALID_QUERY, "type", "text")
        self.test_query(
            input_query=modified_query,
            description="Query Valid Type Parameter in Content",
            assert_type=self.AssertType.PASS,
        )

    # ==================== RESPONSE TESTS ====================

    def test_response(self, input_response, description: str, assert_type: BaseEvaluatorRunner.AssertType):
        """Helper method to test various query inputs."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=input_response,
            tool_calls=None,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, description)

        self.assert_expected_behavior(assert_type, result_data)

    def test_response_not_present(self):
        """Response not present - should raise missing field error."""
        self.test_response(
            input_response=None, description="Response Not Present", assert_type=self.AssertType.MISSING_FIELD
        )

    def test_response_as_string(self):
        """Response as string - should pass."""
        self.test_response(
            input_response=self.STRING_RESPONSE, description="Response String", assert_type=self.AssertType.PASS
        )

    def test_response_invalid_format_as_string(self):
        """Response as string - should pass."""
        self.test_response(
            input_response=self.INVALID_RESPONSE_AS_STRING,
            description="Response Invalid Format as String",
            assert_type=self.AssertType.PASS,
        )

    def test_response_invalid_format(self):
        """Response in invalid format - should raise invalid value error."""
        self.test_response(
            input_response=self.INVALID_RESPONSE,
            description="Response Invalid Format",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_wrong_type(self):
        """Response in wrong type - should raise invalid value error."""
        self.test_response(
            input_response=self.WRONG_TYPE,
            description="Response Wrong Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_empty_list(self):
        """Response as empty list - should raise missing field error."""
        self.test_response(
            input_response=self.EMPTY_LIST,
            description="Response Empty List",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    def test_response_minimal_format(self):
        """Response in minimal format - should pass."""
        self.test_response(
            input_response=self.MINIMAL_RESPONSE,
            description="Response Minimal without Tool Calls",
            assert_type=self.AssertType.PASS,
        )

    # ==================== RESPONSE PARAMETER TESTS ====================

    def test_response_missing_role_parameter(self):
        """Response is missing role parameter - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "role")
        self.test_response(
            input_response=modified_response,
            description="Response Missing Role Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_invalid_role_parameter(self):
        """Response with invalid role parameter - should raise invalid value error."""
        modified_response = self.update_parameter_in_input(self.VALID_RESPONSE, "role", "invalid_role")
        # TODO: Should this be INVALID_VALUE instead of PASS?
        self.test_response(
            input_response=modified_response,
            description="Response Invalid Role Parameter",
            assert_type=self.AssertType.PASS,
        )

    def test_response_missing_content_parameter(self):
        """Response is missing content parameter - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input(self.VALID_RESPONSE, "content")
        self.test_response(
            input_response=modified_response,
            description="Response Missing Content Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_empty_content_parameter(self):
        """Response has empty content parameter - should raise invalid value error."""
        modified_response = self.update_parameter_in_input(self.VALID_RESPONSE, "content", self.EMPTY_LIST)
        self.test_response(
            input_response=modified_response,
            description="Response Empty Content Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_invalid_content_parameter_type(self):
        """Response has invalid content parameter type - should raise invalid value error."""
        modified_response = self.update_parameter_in_input(self.VALID_RESPONSE, "content", self.WRONG_TYPE)
        self.test_response(
            input_response=modified_response,
            description="Response Invalid Content Parameter Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_valid_content_parameter(self):
        """Response has valid content parameter - should pass."""
        modified_response = self.VALID_RESPONSE.copy()
        modified_response = modified_response[-1]["content"] = self.STRING_RESPONSE
        self.test_response(
            input_response=modified_response,
            description="Response Valid Content Parameter",
            assert_type=self.AssertType.PASS,
        )

    def test_response_missing_text_parameter_in_content(self):
        """Response is missing text parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "text")
        self.test_response(
            input_response=modified_response,
            description="Response Missing Text Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_missing_type_parameter_in_content(self):
        """Response is missing type parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "type")
        self.test_response(
            input_response=modified_response,
            description="Response Missing Type Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_invalid_type_parameter_in_content(self):
        """Response has invalid type parameter in content - should raise invalid value error."""
        modified_response = self.VALID_RESPONSE.copy()
        modified_response = modified_response[-1]["content"]["type"] = "invalid_type"
        self.test_response(
            input_response=modified_response,
            description="Response Invalid Type Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_valid_type_parameter_in_content(self):
        """Response has valid type parameter in content - should pass."""
        modified_response = self.VALID_RESPONSE.copy()
        modified_response = modified_response[-1]["content"]["type"] = "text"
        self.test_response(
            input_response=modified_response,
            description="Response Valid Type Parameter in Content",
            assert_type=self.AssertType.PASS,
        )

    def test_response_missing_tool_call_id_parameter(self):
        """Response is missing tool_call_id parameter - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input(self.VALID_RESPONSE, "tool_call_id")
        self.test_response(
            input_response=modified_response,
            description="Response Missing tool_call_id Parameter",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_missing_tool_result_parameter_in_content(self):
        """Response is missing tool_result parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "tool_result")
        self.test_response(
            input_response=modified_response,
            description="Response Missing tool_result Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_invalid_tool_result_parameter_in_content(self):
        """Response has invalid tool_result parameter in content - should raise invalid value error."""
        modified_response = self.update_parameter_in_input_content(self.VALID_RESPONSE, "tool_result", self.WRONG_TYPE)
        self.test_response(
            input_response=modified_response,
            description="Response Invalid tool_result Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_missing_tool_call_id_parameter_in_content(self):
        """Response is missing tool_call_id parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "tool_call_id")
        self.test_response(
            input_response=modified_response,
            description="Response Missing tool_call_id Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_missing_name_parameter_in_content(self):
        """Response is missing name parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "name")
        self.test_response(
            input_response=modified_response,
            description="Response Missing name Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_missing_arguments_parameter_in_content(self):
        """Response is missing arguments parameter in content - should raise invalid value error."""
        modified_response = self.remove_parameter_from_input_content(self.VALID_RESPONSE, "arguments")
        self.test_response(
            input_response=modified_response,
            description="Response Missing arguments Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_response_invalid_arguments_parameter_in_content(self):
        """Response has invalid arguments parameter in content - should raise invalid value error."""
        modified_response = self.update_parameter_in_input_content(self.VALID_RESPONSE, "arguments", self.WRONG_TYPE)
        self.test_response(
            input_response=modified_response,
            description="Response Invalid arguments Parameter in Content",
            assert_type=self.AssertType.INVALID_VALUE,
        )
