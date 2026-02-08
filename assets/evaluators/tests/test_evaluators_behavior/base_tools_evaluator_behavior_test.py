# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of for tools evaluators.

Tests various input scenarios: query, response, and tool_definitions.
"""

import json
from typing import List
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest


class BaseToolsEvaluatorBehaviorTest(BaseEvaluatorBehaviorTest):
    """
    Base class for tools evaluator behavioral tests with tool_definitions.

    Extends BaseEvaluatorBehaviorTest with tool definition support.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "ToolOutputUtilization")
    Subclasses may override:
    - requires_tool_definitions: bool - whether tool definitions are required or optional
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    """

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for tools evaluators."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_details",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]

    # Test Configs
    requires_tool_definitions: bool = False

    # region Test Data
    # Tool definition test data
    VALID_TOOL_DEFINITIONS = [
        {
            "name": "fetch_weather",
            "description": "Fetches the weather information for the specified location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        },
        {
            "name": "send_email",
            "description": "Sends an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
            },
        },
    ]

    INVALID_TOOL_DEFINITIONS = [
        {
            "fetch_weather": {
                "description": "Fetches the weather information for the specified location.",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                },
            }
        },
        {
            "send_email": {
                "description": "Sends an email.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            },
        },
    ]

    INVALID_TOOL_DEFINITIONS_AS_STRING: str = json.dumps(INVALID_TOOL_DEFINITIONS)
    # endregion

    # ==================== TOOL DEFINITIONS TESTS ====================

    def run_tool_definitions_test(
        self, input_tool_definitions, description: str, assert_type: BaseEvaluatorBehaviorTest.AssertType
    ):
        """Test various tool definitions inputs."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=input_tool_definitions,
        )
        result_data = self._extract_and_print_result(results, description)

        expected_behavior = assert_type
        if not self.requires_tool_definitions and assert_type != self.AssertType.INVALID_VALUE:
            expected_behavior = self.AssertType.PASS

        self.assert_expected_behavior(expected_behavior, result_data)

    def test_tool_definitions_not_present(self):
        """Tool definitions not present - should raise missing field error."""
        self.run_tool_definitions_test(
            input_tool_definitions=None,
            description="Tool Definitions Not Present",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    def test_tool_definitions_as_string(self):
        """Tool definitions as string - should pass."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS_AS_STRING,
            description="Tool Definitions String",
            assert_type=self.AssertType.PASS,
        )

    def test_tool_definitions_invalid_format_as_string(self):
        """Tool definitions as string - should pass."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS_AS_STRING,
            description="Tool Definitions Invalid Format as String",
            assert_type=self.AssertType.PASS,
        )

    def test_tool_definitions_invalid_format(self):
        """Tool definitions in invalid format - should raise invalid value error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.INVALID_TOOL_DEFINITIONS,
            description="Tool Definitions Invalid Format",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_wrong_type(self):
        """Tool definitions in wrong type - should raise invalid value error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.WRONG_TYPE,
            description="Tool Definitions Wrong Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_empty_list(self):
        """Tool definitions as empty list - should raise missing field error."""
        self.run_tool_definitions_test(
            input_tool_definitions=self.EMPTY_LIST,
            description="Tool Definitions Empty List",
            assert_type=self.AssertType.MISSING_FIELD,
        )

    # ==================== TOOL DEFINITIONS PARAMETER TESTS ====================
    def test_tool_definitions_missing_name_parameter(self):
        """Tool definitions missing 'name' parameter - should raise invalid value error."""
        modified_tool_definitions = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="name"
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Missing 'name'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_missing_parameters_parameter(self):
        """Tool definitions missing 'parameters' parameter - should raise invalid value error."""
        modified_tool_definitions = self.remove_parameter_from_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="parameters"
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Missing 'parameters'",
            assert_type=self.AssertType.INVALID_VALUE,
        )

    def test_tool_definitions_invalid_parameters_type(self):
        """Tool definitions with invalid 'parameters' type - should raise invalid value error."""
        modified_tool_definitions = self.update_parameter_in_input(
            input_data=self.VALID_TOOL_DEFINITIONS, parameter_name="parameters", parameter_value=self.WRONG_TYPE
        )
        self.run_tool_definitions_test(
            input_tool_definitions=modified_tool_definitions,
            description="Tool Definitions Invalid 'parameters' Type",
            assert_type=self.AssertType.INVALID_VALUE,
        )
