# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Output Utilization Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import ToolOutputUtilizationEvaluator
from .common_test_data import ToolDefinitions


@pytest.mark.quality
class TestToolOutputUtilizationEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Output Utilization Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Tool Output Utilization evaluates whether the agent correctly interprets
    and uses tool outputs when producing responses, considering:
    - Correct reflection of factual content from tool outputs
    - Correct reuse of tool outputs in new tool calls or textual responses
    - Detection of mismatches, fabrications, and misinterpretations
    """

    evaluator_type = ToolOutputUtilizationEvaluator

    # ==================== PASS CASES ====================

    def test_pass_correct_tool_output_usage(self) -> None:
        """Test case: PASS - Agent correctly uses tool output in response."""
        self.run_quality_test(
            test_label="PASS-correct-tool-output-usage",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What's the weather in Seattle?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {"location": "Seattle"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"city": "Seattle", "temp": 65, "condition": "Partly cloudy"}',
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The weather in Seattle is 65°F and partly cloudy."}],
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location. Returns temperature in Fahrenheit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                }
            ],
        )

    def test_pass_correct_account_transfer(self) -> None:
        """Test case: PASS - Agent correctly uses account IDs from tool output in subsequent call."""
        self.run_quality_test(
            test_label="PASS-correct-account-transfer",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Can you transfer $500 from my checking to my savings account?"}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_account_balances",
                            "arguments": {"user_id": "USER456"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"accounts": [{"account_id": "CHK001", "type": "checking", '
                                '"balance": 1250.75}, {"account_id": "SAV001", "type": "savings", '
                                '"balance": 3400.20}]}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You have $1,250.75 in checking and $3,400.20 in savings. "
                                "You have enough for the transfer."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Great, please go ahead."}]},
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "transfer_funds",
                            "arguments": {"from_account": "CHK001", "to_account": "SAV001", "amount": 500},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"transaction_id": "TXN789123", "status": "completed", '
                                '"from_account": "CHK001", "to_account": "SAV001", "amount": 500, '
                                '"new_balances": {"CHK001": 750.75, "SAV001": 3900.20}}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transfer completed successfully. Checking now has $750.75 and savings $3,900.20.",
                        }
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "get_account_balances",
                    "description": "Get account balances for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "User ID"}
                        },
                        "required": ["user_id"],
                    },
                },
                {
                    "name": "transfer_funds",
                    "description": "Transfer funds between accounts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_account": {"type": "string", "description": "Source account ID"},
                            "to_account": {"type": "string", "description": "Destination account ID"},
                            "amount": {"type": "number", "description": "Amount to transfer"},
                        },
                        "required": ["from_account", "to_account", "amount"],
                    },
                },
            ],
        )

    def test_pass_correct_order_details(self) -> None:
        """Test case: PASS - Agent correctly reports order details from tool output."""
        self.run_quality_test(
            test_label="PASS-correct-order-details",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What's the status of my order #12345?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_order_status",
                            "arguments": {"order_id": "12345"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"order_id": "12345", "status": "shipped", '
                                '"tracking_number": "1Z999AA10123456784", '
                                '"estimated_delivery": "2025-02-05"}'
                            ),
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your order #12345 has been shipped. The tracking number is "
                                "1Z999AA10123456784 and the estimated delivery date is February 5, 2025."
                            ),
                        }
                    ],
                }
            ],
            tool_definitions=[ToolDefinitions.GET_ORDER_STATUS],
        )

    def test_pass_multiple_tool_outputs_combined(self) -> None:
        """Test case: PASS - Agent correctly combines information from multiple tool outputs."""
        self.run_quality_test(
            test_label="PASS-multiple-tool-outputs-combined",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Compare the weather in Seattle and Portland"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_weather",
                            "arguments": {"location": "Seattle"},
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "get_weather",
                            "arguments": {"location": "Portland"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"city": "Seattle", "temp": 58, "condition": "Cloudy"}',
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"city": "Portland", "temp": 62, "condition": "Sunny"}',
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Seattle is currently 58°F and cloudy, while Portland is 62°F and sunny. "
                                "Portland is slightly warmer today."
                            ),
                        }
                    ],
                }
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a location. Returns temperature in Fahrenheit.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                }
            ],
        )

    def test_pass_order_status_with_system_message(self) -> None:
        """Test case: PASS - Agent correctly reports order status from tool output with system message."""
        self.run_quality_test(
            test_label="PASS-order-status-with-system-message",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "system",
                    "content": "You are a customer service assistant helping with order inquiries.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the status of order #12345?"}],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_456",
                            "name": "get_order_status",
                            "arguments": {"order_id": "12345"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_456",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"order_id": "12345", "status": "shipped", '
                                '"tracking_number": "1Z999AA1234567890", "estimated_delivery": "2024-10-03"}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your order #12345 has been shipped! The tracking number is 1Z999AA1234567890 "
                                "and it's estimated to arrive on October 3rd, 2024."
                            ),
                        }
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "get_order_status",
                    "description": "Retrieves current status and details for an order",
                    "parameters": {
                        "type": "object",
                        "properties": {"order_id": {"type": "string"}},
                    },
                }
            ],
        )

    # ==================== FAIL CASES ====================

    def test_fail_unit_misinterpretation(self) -> None:
        """Test case: FAIL - Agent incorrectly interprets temperature unit from tool output."""
        self.run_quality_test(
            test_label="FAIL-unit-misinterpretation",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What's the current temperature in Rome?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "weather_api",
                            "arguments": {"city": "Rome"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result", "tool_result": '{"city": "Rome", "temp": 28, "condition": "Sunny"}'}
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "It's currently 28°F and sunny in Rome."}],
                }
            ],
            tool_definitions=[
                {
                    "name": "weather_api",
                    "description": "Gets current weather for a city and returns temperature in Celsius",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                }
            ],
        )

    def test_fail_fabricated_inventory_claim(self) -> None:
        """Test case: FAIL - Agent fabricates claim that item is in stock when tool shows qty=0."""
        self.run_quality_test(
            test_label="FAIL-fabricated-inventory-claim",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Can you check if item B123 is available?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "inventory_api",
                            "arguments": {"item_id": "B123"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {"type": "tool_result", "tool_result": '{"item_id": "B123", "qty": 0, "eta": "2025-10-07"}'}
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The item is in stock till the 7th of October."}],
                }
            ],
            tool_definitions=[
                {
                    "name": "inventory_api",
                    "description": (
                        "Check inventory for an item. Returns quantity available and expected "
                        "availability date if out of stock."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string", "description": "Item ID"}
                        },
                        "required": ["item_id"],
                    },
                }
            ],
        )

    def test_fail_wrong_account_id_in_tool_call(self) -> None:
        """Test case: FAIL - Agent uses wrong account ID from prior tool output in new tool call."""
        self.run_quality_test(
            test_label="FAIL-wrong-account-id-in-tool-call",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Show my latest account transactions."}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_account_list",
                            "arguments": {"user_id": "U123"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"accounts": [{"id": "CHK100", "type": "checking"}, '
                                '{"id": "SAV200", "type": "savings"}]}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You have two accounts: CHK100 (checking) and SAV200 (savings). "
                                "Which one do you want transactions for?"
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Please get the transaction history for my checking account."}
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "get_transactions",
                            "arguments": {"account_id": "SAV200"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {"type": "tool_result", "tool_result": '{"transactions": [{"id": "T1", "amount": 100}]}'}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The latest transactions for your checking account are: T1 - $100."}
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "get_account_list",
                    "description": "Get list of accounts for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string", "description": "User ID"}},
                        "required": ["user_id"],
                    },
                },
                {
                    "name": "get_transactions",
                    "description": "Get transaction history for an account",
                    "parameters": {
                        "type": "object",
                        "properties": {"account_id": {"type": "string", "description": "Account ID"}},
                        "required": ["account_id"],
                    },
                },
            ],
        )

    def test_fail_wrong_value_reported(self) -> None:
        """Test case: FAIL - Agent reports wrong balance value from tool output."""
        self.run_quality_test(
            test_label="FAIL-wrong-value-reported",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What's my account balance?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_balance",
                            "arguments": {"user_id": "USER123"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [{"type": "tool_result", "tool_result": '{"balance": 1523.47, "currency": "USD"}'}],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Your current account balance is $2,523.47."}],
                }
            ],
            tool_definitions=[
                {
                    "name": "get_balance",
                    "description": "Get account balance for a user",
                    "parameters": {
                        "type": "object",
                        "properties": {"user_id": {"type": "string", "description": "User ID"}},
                        "required": ["user_id"],
                    },
                }
            ],
        )

    def test_fail_fabricated_field(self) -> None:
        """Test case: FAIL - Agent fabricates information not present in tool output."""
        self.run_quality_test(
            test_label="FAIL-fabricated-field",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Get details about product SKU-789"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_product",
                            "arguments": {"sku": "SKU-789"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"sku": "SKU-789", "name": "Wireless Mouse", "price": 29.99}',
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Product SKU-789 is a Wireless Mouse priced at $29.99. "
                                "It has a 4.5 star rating and comes with free shipping."
                            ),
                        }
                    ],
                }
            ],
            tool_definitions=[
                {
                    "name": "get_product",
                    "description": "Get product details by SKU",
                    "parameters": {
                        "type": "object",
                        "properties": {"sku": {"type": "string", "description": "Product SKU"}},
                        "required": ["sku"],
                    },
                }
            ],
        )

    def test_fail_swapped_entities(self) -> None:
        """Test case: FAIL - Agent swaps entity values from tool output."""
        self.run_quality_test(
            test_label="FAIL-swapped-entities",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Compare flights from NYC to LA"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "search_flights",
                            "arguments": {"from": "NYC", "to": "LA"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"flights": [{"airline": "United", "price": 350, "duration": "5h 30m"}, '
                                '{"airline": "Delta", "price": 420, "duration": "5h 15m"}]}'
                            ),
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I found two flights. United costs $420 with a 5h 15m flight time, "
                                "and Delta costs $350 with a 5h 30m duration."
                            ),
                        }
                    ],
                }
            ],
            tool_definitions=[
                {
                    "name": "search_flights",
                    "description": "Search for available flights",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string", "description": "Departure city"},
                            "to": {"type": "string", "description": "Destination city"},
                        },
                        "required": ["from", "to"],
                    },
                }
            ],
        )

    def test_fail_misinterprets_status(self) -> None:
        """Test case: FAIL - Agent misinterprets status from tool output."""
        self.run_quality_test(
            test_label="FAIL-misinterprets-status",
            expected=ExpectedResult.FAIL,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Is my payment processed?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "check_payment",
                            "arguments": {"payment_id": "PAY001"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"payment_id": "PAY001", "status": "pending", "amount": 150.00}',
                        }
                    ],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Yes, your payment of $150.00 has been successfully processed!"}
                    ],
                }
            ],
            tool_definitions=[
                {
                    "name": "check_payment",
                    "description": "Check payment status",
                    "parameters": {
                        "type": "object",
                        "properties": {"payment_id": {"type": "string", "description": "Payment ID"}},
                        "required": ["payment_id"],
                    },
                }
            ],
        )

    def test_fail_selective_reporting_medical_results(self) -> None:
        """Test case: FAIL - Agent selectively reports only normal values, ignoring high glucose."""
        self.run_quality_test(
            test_label="FAIL-selective-reporting-medical-results",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "system",
                    "content": "You are a medical assistant helping patients understand their test results.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "What do my latest blood test results show?"}],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_303",
                            "name": "get_lab_results",
                            "arguments": {"patient_id": "PAT123", "test_type": "comprehensive_metabolic_panel"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_303",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"patient_id": "PAT123", "test_date": "2024-09-25", '
                                '"results": {"glucose": {"value": 180, "unit": "mg/dL", '
                                '"reference_range": "70-100", "status": "HIGH"}, '
                                '"cholesterol": {"value": 95, "unit": "mg/dL", '
                                '"reference_range": "less than 200", "status": "NORMAL"}}}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your blood test results from September 25th show that your cholesterol level is "
                                "95 mg/dL, which is in the normal range (should be less than 200). "
                                "Everything looks good!"
                            ),
                        }
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "get_lab_results",
                    "description": "Retrieves laboratory test results for a patient",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "test_type": {"type": "string"},
                        },
                    },
                }
            ],
        )

    def test_fail_fabricates_after_tool_error(self) -> None:
        """Test case: FAIL - Agent fabricates availability info after tool returns error."""
        self.run_quality_test(
            test_label="FAIL-fabricates-after-tool-error",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "system",
                    "content": "You are an e-commerce assistant helping customers check product availability.",
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Is the blue iPhone 15 Pro in stock?"}],
                },
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_101",
                            "name": "check_inventory",
                            "arguments": {"product_id": "iphone15pro-blue", "color": "blue"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_101",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"error": "Error executing function \'check_inventory\': '
                                'Network timeout while accessing inventory service"}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Yes, the blue iPhone 15 Pro is currently available but in limited stock. "
                                "Would you like me to help you place an order before it sells out?"
                            ),
                        }
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "check_inventory",
                    "description": "Checks current inventory status for a product",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "color": {"type": "string"},
                        },
                    },
                }
            ],
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_paraphrased_response(self) -> None:
        """Test case: EDGE - Agent paraphrases tool output correctly (should pass)."""
        self.run_quality_test(
            test_label="EDGE-paraphrased-response",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What time does the store close?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_store_hours",
                            "arguments": {"store_id": "STORE001"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"store_id": "STORE001", "open": "09:00", "close": "21:00"}',
                        }
                    ],
                },
            ],
            response=[{"role": "assistant", "content": [{"type": "text", "text": "The store closes at 9 PM."}]}],
            tool_definitions=[
                {
                    "name": "get_store_hours",
                    "description": "Get store operating hours",
                    "parameters": {
                        "type": "object",
                        "properties": {"store_id": {"type": "string", "description": "Store ID"}},
                        "required": ["store_id"],
                    },
                }
            ],
        )

    def test_edge_case_partial_information(self) -> None:
        """Test case: EDGE - Agent uses only relevant parts of tool output (should pass)."""
        self.run_quality_test(
            test_label="EDGE-partial-information",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "What's the price of the laptop?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_product",
                            "arguments": {"product_id": "LAPTOP001"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"product_id": "LAPTOP001", "name": "ProBook 15", "price": 999.99, '
                                '"weight": "2.1kg", "color": "Silver", "warranty": "2 years"}'
                            ),
                        }
                    ],
                },
            ],
            response=[{"role": "assistant", "content": [{"type": "text", "text": "The laptop costs $999.99."}]}],
            tool_definitions=[
                {
                    "name": "get_product",
                    "description": "Get product details",
                    "parameters": {
                        "type": "object",
                        "properties": {"product_id": {"type": "string", "description": "Product ID"}},
                        "required": ["product_id"],
                    },
                }
            ],
        )

    def test_edge_case_multiple_turns_correct(self) -> None:
        """Test case: EDGE - Multi-turn conversation with correct tool output usage throughout."""
        self.run_quality_test(
            test_label="EDGE-multiple-turns-correct",
            expected=ExpectedResult.PASS,
            query=[
                {"role": "user", "content": [{"type": "text", "text": "Search for hotels in Paris"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "search_hotels",
                            "arguments": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": (
                                '{"hotels": [{"id": "H1", "name": "Grand Paris", "price": 200}, '
                                '{"id": "H2", "name": "Eiffel View", "price": 350}]}'
                            ),
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I found two hotels: Grand Paris at $200/night and Eiffel View at $350/night.",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Book the cheaper one"}]},
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "book_hotel",
                            "arguments": {"hotel_id": "H1"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"confirmation": "CONF123", "hotel": "Grand Paris", "total": 200}',
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I've booked the Grand Paris for you. Your confirmation number is "
                                "CONF123 and the total is $200."
                            ),
                        }
                    ],
                },
            ],
            tool_definitions=[
                {
                    "name": "search_hotels",
                    "description": "Search for hotels in a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string", "description": "City name"}},
                        "required": ["city"],
                    },
                },
                {
                    "name": "book_hotel",
                    "description": "Book a hotel",
                    "parameters": {
                        "type": "object",
                        "properties": {"hotel_id": {"type": "string", "description": "Hotel ID"}},
                        "required": ["hotel_id"],
                    },
                },
            ],
        )
