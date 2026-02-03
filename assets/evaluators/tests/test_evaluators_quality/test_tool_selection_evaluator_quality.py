# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Tool Selection Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.tool_selection.evaluator._tool_selection import ToolSelectionEvaluator
from .common_test_data import ToolDefinitions, ToolDefinitionSets


@pytest.mark.quality
class TestToolSelectionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Tool Selection Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Tool Selection evaluates whether the agent selected the appropriate tools
    for the given task, considering factors like:
    - Selecting the right tool for the job
    - Avoiding unnecessary/irrelevant tool calls
    - Including all necessary tools for complex tasks
    """

    evaluator_type = ToolSelectionEvaluator

    # ==================== PASS CASES ====================

    def test_pass_single_correct_tool(self) -> None:
        """Test case: PASS - Correct tool selected for single-tool task."""
        self.run_quality_test(
            test_label="PASS-single-correct-tool",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Send an email to john@example.com with subject 'Meeting'",
                        }
                    ],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "send_email",
                            "arguments": {
                                "to": "john@example.com",
                                "subject": "Meeting",
                            },
                        }
                    ],
                }
            ],
            tool_definitions=ToolDefinitionSets.EMAIL_AND_FILE,
        )

    # ==================== FAIL CASES ====================

    def test_fail_single_irrelevant_tool(self) -> None:
        """Test case: FAIL - Irrelevant tool selected instead of correct one."""
        self.run_quality_test(
            test_label="FAIL-single-irrelevant-tool",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Send an email to john@example.com",
                        }
                    ],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "delete_file",
                            "arguments": {
                                "filename": "email.txt"
                            },
                        }
                    ],
                }
            ],
            tool_definitions=ToolDefinitionSets.EMAIL_AND_FILE,
        )

    def test_fail_missing_calls(self) -> None:
        """Test case: FAIL - Only part of required tools are called."""
        self.run_quality_test(
            test_label="FAIL-missing-calls",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Search for iPhone 15 and add it to my cart"}],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "product_search",
                            "arguments": {"query": "iPhone 15"},
                        }
                    ],
                }
            ],
            tool_definitions=ToolDefinitionSets.SHOPPING,
        )

    def test_pass_duplicate_calls(self) -> None:
        """
        Test case: PASS - Same tool called multiple times unnecessarily.
        For Tool Selection, redundant calls are a pass but missing calls are a fail.
        """
        self.run_quality_test(
            test_label="PASS-duplicate-calls",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Send email to john@example.com"}],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "send_email",
                            "arguments": {"to": "john@example.com"},
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "send_email",
                            "arguments": {"to": "john@example.com"},
                        },
                    ],
                }
            ],
            tool_definitions=[ToolDefinitions.SEND_EMAIL_BASIC],
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_tool_calls_in_history(self) -> None:
        """Test case: EDGE - Necessary tools already called in conversation history."""
        self.run_quality_test(
            test_label="EDGE-tool-calls-in-history",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Calculate total for items in my cart",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "fetch_items_in_cart",
                            "arguments": {},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Items: iPhone ($999), Case ($49)",
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "You have iPhone and Case in your cart.",
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "calculate_total",
                            "arguments": {},
                        },
                    ],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "calculate_total",
                            "arguments": {},
                        }
                    ],
                }
            ],
            tool_definitions=ToolDefinitionSets.CART_OPERATIONS,
        )

    def test_edge_case_complete_but_excessive(self) -> None:
        """Test case: EDGE - All necessary tools called plus some unnecessary ones."""
        self.run_quality_test(
            test_label="EDGE-complete-but-excessive",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Process payment for my order"}],
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_call", "tool_call_id": "call_1", "name": "validate_card", "arguments": {}},
                        {"type": "tool_call", "tool_call_id": "call_2", "name": "check_balance", "arguments": {}},
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_3",
                            "name": "process_transaction",
                            "arguments": {},
                        },
                        {"type": "tool_call", "tool_call_id": "call_4", "name": "send_receipt", "arguments": {}},
                        {"type": "tool_call", "tool_call_id": "call_5", "name": "log_transaction", "arguments": {}},
                        {"type": "tool_call", "tool_call_id": "call_6", "name": "notify_user", "arguments": {}},
                    ],
                }
            ],
            tool_definitions=ToolDefinitionSets.PAYMENT_PROCESSING,
        )
