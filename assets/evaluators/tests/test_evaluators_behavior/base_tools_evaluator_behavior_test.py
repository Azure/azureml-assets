# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for behavioral tests of for tools evaluators.

Tests various input scenarios: query, response, and tool_definitions.
"""

import json
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ..common.evaluator_mock_config import (
    create_none_score_flow_side_effect,
    assert_none_score_result,
)


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
    - expected_result_fields: list - expected fields in the evaluation result
      tools evaluators
    """

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

    def run_skipped_llm_status_not_applicable_test(self):
        """Run a skipped-status flow output and assert a not-applicable result.

        Regression: when the LLM flow returns ``status='skipped'`` (score=None), the
        evaluator must return a standardized not-applicable result without crashing.
        """
        self._run_none_score_not_applicable_test(self.VALID_RESPONSE)

    def run_intermediate_response_not_applicable_test(self):
        """Run an intermediate (function_call) response and assert a not-applicable result.

        Regression: a response whose final assistant turn is an unresolved function_call
        must be treated as not-applicable rather than evaluated.
        """
        self._run_none_score_not_applicable_test(self.FUNCTION_CALL_ONLY_RESPONSE)

    def _run_none_score_not_applicable_test(self, response):
        """Mock the flow to a None/skipped score, run with ``response``, assert not-applicable.

        Shared by the skipped-status and intermediate-response regressions, which differ
        only in the ``response`` payload passed to the evaluator.
        """
        result = self._run_evaluation_with_flow_side_effect(
            create_none_score_flow_side_effect(),
            query=self.VALID_QUERY,
            response=response,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        assert_none_score_result(result, self.result_key)

    def test_util_tool_definitions_reach_flow_e2e(self):
        """End-to-end: needed tool definitions are extracted (built-in + provided) and reach the flow."""
        _, captured = self._run_and_capture_flow_input(
            query=self.VALID_QUERY,
            response=self.VALID_RESPONSE,
            tool_calls=self.VALID_TOOL_CALLS,
            tool_definitions=self.VALID_TOOL_DEFINITIONS,
        )
        flow_input_json = json.dumps(captured, default=str)
        assert (
            "fetch_weather" in flow_input_json or "send_email" in flow_input_json
        ), "tool definitions did not reach the flow"

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
