# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Input Accuracy Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import ToolInputAccuracyEvaluator
from .common_test_data import ToolDefinitions, ToolDefinitionSets


@pytest.mark.quality
class TestToolInputAccuracyEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Input Accuracy Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolInputAccuracyEvaluator

    def test_pass_single_call_correct(self) -> None:
        """Test case: PASS - Correct tool call with all parameters properly specified."""
        self.run_quality_test(
            test_label="PASS-single-call-correct",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Search for iPhone 15"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": "iPhone 15"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.PRODUCT_SEARCH],
        )

    def test_pass_correct_format(self) -> None:
        """Test case: PASS - Date parameter in correct YYYY-MM-DD format."""
        self.run_quality_test(
            test_label="PASS-correct-format",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Book appointment for January 5, 2025"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "book_appointment", "arguments": {"date": "2025-01-05"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.BOOK_APPOINTMENT],
        )

    def test_fail_incorrect_format(self) -> None:
        """Test case: FAIL - Date in wrong format (YYYY-MM-DD when DD-MM-YYYY expected)."""
        self.run_quality_test(
            test_label="FAIL-incorrect-format",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Book appointment for January 5, 2025"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "book_appointment", "arguments": {"date": "2025-01-05"}}
                ]}
            ],
            tool_definitions=[
                {
                    "name": "book_appointment",
                    "description": "Book an appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "description": "Date in DD-MM-YYYY format"}
                        },
                        "required": ["date"]
                    }
                }
            ],
        )

    def test_fail_missing_parameter(self) -> None:
        """Test case: FAIL - Missing required subject parameter in email tool call."""
        self.run_quality_test(
            test_label="FAIL-missing-parameter",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Send email to john@example.com with subject 'Hello'"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "send_email", "arguments": {"to": "john@example.com"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.SEND_EMAIL_BASIC],
        )

    def test_fail_extra_parameter(self) -> None:
        """Test case: FAIL - Unexpected extra parameter provided in tool call."""
        self.run_quality_test(
            test_label="FAIL-extra-parameter",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Get server time"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "get_server_time", "arguments": {"timezone": "PST", "format": "12h"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.GET_SERVER_TIME],
        )

    def test_fail_param_not_grounded_from_query(self) -> None:
        """Test case: FAIL - Parameter value not grounded in user query (Samsung vs iPhone)."""
        self.run_quality_test(
            test_label="FAIL-param-not-grounded-from-query",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Search for iPhone 15"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": "Samsung Galaxy"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.PRODUCT_SEARCH],
        )

    def test_fail_param_not_grounded_from_tool_result(self) -> None:
        """Test case: FAIL - Parameter value not grounded in previous tool results (wrong coordinates)."""
        self.run_quality_test(
            test_label="FAIL-param-not-grounded-from-tool-result",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Calculate distance between Seattle and Portland"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "get_coordinates", "arguments": {"city": "Seattle"}},
                    {"type": "tool_call", "tool_call_id": "call_2", "name": "get_coordinates", "arguments": {"city": "Portland"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "tool_result", "tool_result": "1, 2"}]},
                {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "tool_result", "tool_result": "3, 4"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_3", "name": "calculate_distance", "arguments": {"lat1": 1, "lon1": 1, "lat2": 3, "lon2": 4}}
                ]}
            ],
            tool_definitions=ToolDefinitionSets.COORDINATES_AND_DISTANCE,
        )

    def test_pass_param_grounded_from_tool_result(self) -> None:
        """Test case: PASS - Parameter correctly grounded from previous tool result."""
        # TODO: Test fails - evaluator fails due to type mismatch (string vs number).
        # Reason: "The tool definition for 'calculate_distance' specifies lat1, lon1, lat2, lon2 must be of type 'number'.
        # However, the parameters passed are strings (e.g., lat1='1' is a string, not a number).
        # This fails the type compliance criterion."
        # Decision needed: Fix test to pass numeric values instead of string representations,
        # or update the test input format to match expected types.
        self.run_quality_test(
            test_label="PASS-param-grounded-from-tool-result",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Calculate distance between Seattle and Portland"}]},
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "get_coordinates", "arguments": {"city": "Seattle"}},
                    {"type": "tool_call", "tool_call_id": "call_2", "name": "get_coordinates", "arguments": {"city": "Portland"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "tool_result", "tool_result": "1, 2"}]},
                {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "tool_result", "tool_result": "3, 4"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_3", "name": "calculate_distance", "arguments": {"lat1": 1, "lon1": 2, "lat2": 3, "lon2": 4}}
                ]}
            ],
            tool_definitions=ToolDefinitionSets.COORDINATES_AND_DISTANCE,
        )

    def test_fail_param_hallucinated(self) -> None:
        """Test case: FAIL - Hallucinated order_id that wasn't mentioned in query."""
        self.run_quality_test(
            test_label="FAIL-param-hallucinated",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Show me my recent orders"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "get_order", "arguments": {"order_id": "789"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.GET_ORDER, ToolDefinitions.GET_RECENT_ORDERS],
        )

    def test_fail_incorrect_type(self) -> None:
        """Test case: FAIL - user_id as string when integer required."""
        self.run_quality_test(
            test_label="FAIL-incorrect-type",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Get user with ID 123"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "get_user", "arguments": {"user_id": "123"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.GET_USER],
        )

    def test_fail_multiple_calls_missing_param(self) -> None:
        """Test case: FAIL - Two email calls, both missing required subject parameter."""
        self.run_quality_test(
            test_label="FAIL-multiple-calls-missing-param",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Send email to john@example.com and sarah@example.com"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "send_email", "arguments": {"to": "john@example.com"}},
                    {"type": "tool_call", "tool_call_id": "call_2", "name": "send_email", "arguments": {"to": "sarah@example.com"}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.SEND_EMAIL_BASIC],
        )

    def test_fail_incorrectly_passed_none(self) -> None:
        """Test case: FAIL - Query parameter incorrectly set to null."""
        self.run_quality_test(
            test_label="FAIL-incorrectly-passed-none",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Search for products"}]}
            ],
            response=[
                {"role": "assistant", "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": None}}
                ]}
            ],
            tool_definitions=[ToolDefinitions.PRODUCT_SEARCH],
        )
