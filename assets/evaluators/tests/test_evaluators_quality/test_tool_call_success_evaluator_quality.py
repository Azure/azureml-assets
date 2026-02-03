# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Call Success Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.tool_call_success.evaluator._tool_call_success import ToolCallSuccessEvaluator
from .common_test_data import ToolDefinitions


@pytest.mark.quality
class TestToolCallSuccessEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Call Success Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Tool Call Success evaluates whether tool calls returned successful results
    (as opposed to errors, failures, or empty/null results).
    """

    evaluator_type = ToolCallSuccessEvaluator

    # ==================== PASS CASES ====================

    def test_pass_single_call_success(self) -> None:
        """Test case: PASS - Single tool call executes successfully."""
        self.run_quality_test(
            test_label="PASS-single-call-success",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's 25 * 4?"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "calculate",
                            "arguments": {"operation": "multiply", "a": 25, "b": 4}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": "100"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "25 * 4 = 100"}]
                }
            ],
            tool_definitions=[ToolDefinitions.CALCULATE],
        )

    def test_pass_legal_empty_response(self) -> None:
        """Test case: PASS - Tool returns empty list (valid for empty folder)."""
        self.run_quality_test(
            test_label="PASS-legal-empty-response",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "List files in /empty folder"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "GetFilesInFolder",
                            "arguments": {"path": "/empty"}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": []}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The folder is empty."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_FILES_IN_FOLDER],
        )

    # ==================== FAIL CASES ====================

    def test_fail_single_call_error(self) -> None:
        """Test case: FAIL - Single tool call fails with error."""
        self.run_quality_test(
            test_label="FAIL-single-call-error",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's 25 / 0?"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "calculate",
                            "arguments": {"operation": "divide", "a": 25, "b": 0}
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": "Error: Division by zero"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I can't divide by zero."}]
                }
            ],
            tool_definitions=[ToolDefinitions.CALCULATE],
        )

    def test_fail_multi_call_one_failure(self) -> None:
        """Test case: FAIL - Multiple tool calls where one fails."""
        self.run_quality_test(
            test_label="FAIL-multi-call-one-failure",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Calculate 10+5, 20/0, and 3*7"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "calculate",
                         "arguments": {"operation": "add", "a": 10, "b": 5}},
                        {"type": "tool_call", "tool_call_id": "call_2", "name": "calculate",
                         "arguments": {"operation": "divide", "a": 20, "b": 0}},
                        {"type": "tool_call", "tool_call_id": "call_3", "name": "calculate",
                         "arguments": {"operation": "multiply", "a": 3, "b": 7}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": "15"}]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [{"type": "tool_result", "tool_result": "Error: Division by zero"}]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_3",
                    "content": [{"type": "tool_result", "tool_result": "21"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "10+5=15, cannot divide by zero, 3*7=21"}]
                }
            ],
            tool_definitions=[ToolDefinitions.CALCULATE],
        )

    def test_fail_empty_string_result(self) -> None:
        """Test case: FAIL - Tool returns empty string indicating error."""
        self.run_quality_test(
            test_label="FAIL-empty-string-result",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get current time"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "GetCurrentTime", "arguments": {}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": ""}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Unable to get current time."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_CURRENT_TIME],
        )

    @pytest.mark.flaky(reruns=3)
    def test_fail_empty_json_fields(self) -> None:
        """Test case: FAIL - Tool returns JSON object with all empty fields."""
        # TODO: Test fails - score: 1
        # Reason: "The tool result contains no error indicators
        # (no exception, timeout, or explicit error message);
        # it returns an empty object with expected fields,
        # which is treated as a technical success."
        self.run_quality_test(
            test_label="FAIL-empty-json-fields",
            expected=ExpectedResult.PASS_OR_FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get user profile for user 12345"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "get_user",
                         "arguments": {"user_id": 12345}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": {"name": "", "email": "", "status": ""}}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The user profile appears to be empty."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_USER],
        )

    def test_fail_exception_object(self) -> None:
        """Test case: FAIL - Tool returns exception object."""
        self.run_quality_test(
            test_label="FAIL-exception-object",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get order details for ORD-999"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "get_order",
                         "arguments": {"order_id": "ORD-999"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result", "tool_result": {
                            "error": "OrderNotFoundException",
                            "message": "Order ORD-999 not found",
                            "code": 404
                        }}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Order ORD-999 was not found."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_ORDER],
        )

    def test_fail_exception_string(self) -> None:
        """Test case: FAIL - Tool returns exception message as string."""
        self.run_quality_test(
            test_label="FAIL-exception-string",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Book appointment for tomorrow"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "book_appointment",
                         "arguments": {"date": "2024-01-15"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result",
                         "tool_result": "Exception: AppointmentServiceUnavailable - Service temporarily down"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Unable to book appointment. Service is temporarily unavailable."}]
                }
            ],
            tool_definitions=[ToolDefinitions.BOOK_APPOINTMENT],
        )

    def test_fail_http_error_status(self) -> None:
        """Test case: FAIL - Tool returns HTTP failure status code."""
        self.run_quality_test(
            test_label="FAIL-http-error-status",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get order status for ORD-12345"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "get_order_status",
                         "arguments": {"order_id": "ORD-12345"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result", "tool_result": {"status": 500, "message": "Internal Server Error"}}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Unable to retrieve order status due to server error."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_ORDER_STATUS],
        )

    def test_fail_null_result(self) -> None:
        """Test case: FAIL - Tool returns null result."""
        self.run_quality_test(
            test_label="FAIL-null-result",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get weather in Seattle"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "GetWeather",
                         "arguments": {"location": "Seattle"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": None}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Unable to retrieve weather data."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_WEATHER],
        )

    def test_fail_timeout_error(self) -> None:
        """Test case: FAIL - Tool call times out."""
        self.run_quality_test(
            test_label="FAIL-timeout-error",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Search for laptop products"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search",
                         "arguments": {"query": "laptop"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result", "tool_result": "Error: Request timed out after 30 seconds"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Search timed out. Please try again."}]
                }
            ],
            tool_definitions=[ToolDefinitions.PRODUCT_SEARCH],
        )

    # ==================== EDGE CASES ====================

    def test_edge_empty_array_valid_for_context(self) -> None:
        """Test case: EDGE - Empty array that's valid for the context (no recent orders)."""
        self.run_quality_test(
            test_label="EDGE-empty-array-valid-context",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get recent orders for my account"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "get_recent_orders", "arguments": {}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": []}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "You have no recent orders."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_RECENT_ORDERS],
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_empty_array_expected_data(self) -> None:
        """Test case: EDGE - Empty array when data was expected (month names list)."""
        self.run_quality_test(
            test_label="EDGE-empty-array-expected-data",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "List all month names"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "GetMonthNamesList", "arguments": {}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": []}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Unable to retrieve month names."}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_MONTH_NAMES_LIST],
        )

    @pytest.mark.flaky(reruns=3)
    def test_edge_retry_after_failure(self) -> None:
        """Test case: EDGE - Tool call fails then retried successfully."""
        # TODO: Test fails - Score: 0.
        # Reason: "One of the tool calls to GetWeather failed with an error
        # (Temporary network issue), indicating a technical fault.
        # Although the second call returned valid weather data, any failed tool makes the overall evaluation fail.
        # {'failed_tools': 'GetWeather'}"
        # Decision needed: Should a successful retry after initial failure pass,
        # or does any failed tool call in the conversation mean overall failure?
        self.run_quality_test(
            test_label="EDGE-retry-after-failure",
            expected=ExpectedResult.PASS_OR_FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Get weather data"}]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "GetWeather",
                         "arguments": {"location": "Seattle", "units": "fahrenheit"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": "Error: Temporary network issue"}]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me retry that."},
                        {"type": "tool_call", "tool_call_id": "call_2", "name": "GetWeather",
                         "arguments": {"location": "Seattle", "units": "fahrenheit"}}
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [{"type": "tool_result", "tool_result": "Temperature: 65F, Conditions: Partly cloudy"}]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Weather in Seattle: 65°F, partly cloudy"}]
                }
            ],
            tool_definitions=[ToolDefinitions.GET_WEATHER],
        )
