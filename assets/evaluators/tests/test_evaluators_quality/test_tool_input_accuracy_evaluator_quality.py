# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Input Accuracy Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import ToolInputAccuracyEvaluator


@pytest.mark.quality
class TestToolInputAccuracyEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Input Accuracy Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = ToolInputAccuracyEvaluator

    def test_single_call_correct(self):
        """Test a simple correct tool call with all parameters properly specified."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Search for iPhone 15"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": "iPhone 15"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "product_search",
                "description": "Search for products",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Single call - correct")
        self.assert_pass(result_data)

    def test_correct_format(self):
        """Test date parameter in correct YYYY-MM-DD format."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Book appointment for January 5, 2025"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "book_appointment", "arguments": {"date": "2025-01-05"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "book_appointment",
                "description": "Book an appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string", "format": "date", "description": "Date in YYYY-MM-DD format"}
                    },
                    "required": ["date"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Correct Format")
        self.assert_pass(result_data)

    def test_incorrect_format(self):
        """Test date in wrong format (YYYY-MM-DD when DD-MM-YYYY expected)."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Book appointment for January 5, 2025"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "book_appointment", "arguments": {"date": "2025-01-05"}}
            ]}
        ]
        tool_definitions = [
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
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Incorrect Format")
        self.assert_fail(result_data)

    def test_missing_parameter(self):
        """Test missing required subject parameter in email tool call."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Send email to john@example.com with subject 'Hello'"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "send_email", "arguments": {"to": "john@example.com"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"}
                    },
                    "required": ["to", "subject"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Missing parameter")
        self.assert_fail(result_data)

    def test_extra_parameter(self):
        """Test unexpected extra parameter provided in tool call."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Get server time"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "get_server_time", "arguments": {"timezone": "PST", "format": "12h"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "get_server_time",
                "description": "Get current server time",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Extra parameter")
        self.assert_fail(result_data)

    def test_parameter_not_grounded_from_user_query(self):
        """Test parameter value not grounded in user query (Samsung vs iPhone)."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Search for iPhone 15"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": "Samsung Galaxy"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "product_search",
                "description": "Search for products",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Parameter not grounded from user query")
        self.assert_fail(result_data)

    def test_parameter_not_grounded_from_previous_tool_result(self):
        """Test parameter value not grounded in previous tool results (wrong coordinates)."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Calculate distance between Seattle and Portland"}]},
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "get_coordinates", "arguments": {"city": "Seattle"}},
                {"type": "tool_call", "tool_call_id": "call_2", "name": "get_coordinates", "arguments": {"city": "Portland"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "tool_result", "tool_result": "1, 2"}]},
            {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "tool_result", "tool_result": "3, 4"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_3", "name": "calculate_distance", "arguments": {"lat1": 1, "lon1": 1, "lat2": 3, "lon2": 4}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "get_coordinates",
                "description": "Get coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            },
            {
                "name": "calculate_distance",
                "description": "Calculate distance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat1": {"type": "number"},
                        "lon1": {"type": "number"},
                        "lat2": {"type": "number"},
                        "lon2": {"type": "number"}
                    },
                    "required": ["lat1", "lon1", "lat2", "lon2"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Parameter not grounded from previous tool result")
        self.assert_fail(result_data)

    def test_parameter_grounded_from_previous_tool_result(self):
        """Test parameter correctly grounded from previous tool result."""
        # TODO: Test fails - evaluator fails due to type mismatch (string vs number).
        # Reason: "The tool definition for 'calculate_distance' specifies lat1, lon1, lat2, lon2 must be of type 'number'.
        # However, the parameters passed are strings (e.g., lat1='1' is a string, not a number).
        # This fails the type compliance criterion."
        # Decision needed: Fix test to pass numeric values instead of string representations,
        # or update the test input format to match expected types.
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Calculate distance between Seattle and Portland"}]},
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "get_coordinates", "arguments": {"city": "Seattle"}},
                {"type": "tool_call", "tool_call_id": "call_2", "name": "get_coordinates", "arguments": {"city": "Portland"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "tool_result", "tool_result": "1, 2"}]},
            {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "tool_result", "tool_result": "3, 4"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_3", "name": "calculate_distance", "arguments": {"lat1": 1, "lon1": 2, "lat2": 3, "lon2": 4}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "get_coordinates",
                "description": "Get coordinates",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            },
            {
                "name": "calculate_distance",
                "description": "Calculate distance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lat1": {"type": "number"},
                        "lon1": {"type": "number"},
                        "lat2": {"type": "number"},
                        "lon2": {"type": "number"}
                    },
                    "required": ["lat1", "lon1", "lat2", "lon2"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Parameter grounded from previous tool result")
        self.assert_pass(result_data)

    def test_parameter_hallucinated(self):
        """Test hallucinated order_id that wasn't mentioned in query."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Show me my recent orders"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "get_order", "arguments": {"order_id": "789"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "get_order",
                "description": "Get order details",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]
                }
            },
            {
                "name": "get_recent_orders",
                "description": "Get recent orders",
                "parameters": {"type": "object", "properties": {}}
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Parameter hallucinated")
        self.assert_fail(result_data)

    def test_incorrect_type(self):
        """Test user_id as string when integer required."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Get user with ID 123"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "get_user", "arguments": {"user_id": "123"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "get_user",
                "description": "Get user by ID",
                "parameters": {
                    "type": "object",
                    "properties": {"user_id": {"type": "integer"}},
                    "required": ["user_id"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Incorrect type")
        self.assert_fail(result_data)

    def test_multiple_tool_calls_missing_parameter(self):
        """Test two email calls, both missing required subject parameter."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Send email to john@example.com and sarah@example.com"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "send_email", "arguments": {"to": "john@example.com"}},
                {"type": "tool_call", "tool_call_id": "call_2", "name": "send_email", "arguments": {"to": "sarah@example.com"}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"}
                    },
                    "required": ["to", "subject"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Multiple tool calls - missing parameter")
        self.assert_fail(result_data)

    def test_incorrectly_passed_none(self):
        """Test query parameter incorrectly set to null."""
        query = [
            {"role": "user", "content": [{"type": "text", "text": "Search for products"}]}
        ]
        response = [
            {"role": "assistant", "content": [
                {"type": "tool_call", "tool_call_id": "call_1", "name": "product_search", "arguments": {"query": None}}
            ]}
        ]
        tool_definitions = [
            {
                "name": "product_search",
                "description": "Search for products",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]

        result = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(result, "Incorrectly passed None")
        self.assert_fail(result_data)
