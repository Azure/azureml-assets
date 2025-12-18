# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of for tools evaluators.

Tests various input scenarios: query, response, and tool_definitions.
"""

from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest


class BaseToolsEvaluatorBehaviorTest(BaseEvaluatorBehaviorTest):
    """
    Base class for tools evaluator behavioral tests with tool_definitions.

    Extends BaseEvaluatorBehaviorTest with tool definition support.
    Subclasses should implement:
    - evaluator_type: type[PromptyEvaluatorBase] - type of the evaluator (e.g., "ToolOutputUtilization")
    Subclasses may override:
    - requires_tool_definitions: bool - whether tool definitions are required
    - requires_query: bool - whether query is required
    - MINIMAL_RESPONSE: list - minimal valid response format for the evaluator
    """

    # Test Configs
    requires_tool_definitions: bool = False
    tools_evaluator: bool = True

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

    # ==================== TOOL DEFINITIONS TESTS ====================

    def test_tool_definitions_not_present(self):
        """Tool definitions not present - should return not_applicable."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=None,
        )
        result_data = self._extract_and_print_result(results, "Tool Definitions Not Present")

        if self.requires_tool_definitions:
            self.assert_error(result_data)
        else:
            self.assert_pass(result_data)

    def test_tool_definitions_invalid_format(self):
        """Tool definitions in invalid format - should return not_applicable."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.INVALID_TOOL_DEFINITIONS,
        )
        result_data = self._extract_and_print_result(results, "Tool Definitions Invalid")

        if self.requires_valid_format:
            self.assert_error(result_data)
        else:
            self.assert_pass(result_data)
