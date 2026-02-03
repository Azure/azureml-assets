# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Task Completion Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from .common_test_data import (
    ToolDefinitions,
    ToolDefinitionSets,
    create_user_message,
    create_assistant_text_message,
    create_tool_call,
    create_assistant_tool_call_message,
    create_tool_result_message,
)


@pytest.mark.quality
class TestTaskCompletionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Task Completion Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Task Completion Scoring Rubric (from prompty):
    - Score 5: Fully completed - all requirements met, excellent execution
    - Score 4: Mostly completed - core requirements met, minor gaps
    - Score 3: Partially completed - some requirements met, significant gaps
    - Score 2: Attempted - tried but failed to complete
    - Score 1: Not addressed - task ignored or completely wrong
    """

    evaluator_type = TaskCompletionEvaluator

    # ==================== PASS CASES ====================

    def test_pass_task_fully_completed(self) -> None:
        """Test case: PASS - Task completed successfully with correct solution."""
        self.run_quality_test(
            test_label="PASS-task-fully-completed",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Fix my Python code won't run: def greet()\n    print('Hello')"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I found the issue - you're missing a colon after the function definition. Here's the corrected code:\n\ndef greet():\n    print('Hello')"
                        }
                    ]
                }
            ],
            tool_definitions=[],
        )

    # ==================== FAIL CASES ====================

    def test_fail_minor_incompletion(self) -> None:
        """Test case: FAIL - Task mostly completed but missing seat numbers in email."""
        self.run_quality_test(
            test_label="FAIL-minor-incompletion",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Book a round-trip flight from NYC to LA on March 15-20 and email me the confirmation with seat numbers"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "book_flight",
                            "arguments": {
                                "from": "NYC",
                                "to": "LA",
                                "departure_date": "2026-03-15",
                                "return_date": "2026-03-20",
                                "trip_type": "round-trip"
                            }
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "send_email",
                            "arguments": {
                                "subject": "Flight Confirmation",
                                "body": "Your round-trip flight is confirmed! NYC to LA departing March 15 at 9:00 AM, returning March 20 at 6:00 PM. Confirmation #FL8392"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Flight booked successfully. Confirmation #FL8392"
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Email sent"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your round-trip flight is confirmed! NYC to LA departing March 15 at 9:00 AM, returning March 20 at 6:00 PM. Confirmation #FL8392 sent to your email."
                        }
                    ]
                }
            ],
            tool_definitions=ToolDefinitionSets.FLIGHT_BOOKING,
        )

    def test_fail_major_incompletion(self) -> None:
        """Test case: FAIL - Task partially completed but missing explanation of Q3 dip."""
        self.run_quality_test(
            test_label="FAIL-major-incompletion",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Show me quarterly revenue trends and explain the Q3 dip"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "get_revenue_data",
                            "arguments": {
                                "period": "quarterly"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Q1: $2.4M, Q2: $2.8M, Q3: $2.1M, Q4: $3.1M"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Quarterly Revenue:\nQ1: $2.4M\nQ2: $2.8M\nQ3: $2.1M\nQ4: $3.1M"
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "get_revenue_data",
                    "description": "Get revenue data",
                    "parameters": {
                        "type": "object",
                        "properties": {"period": {"type": "string"}},
                        "required": ["period"]
                    }
                }
            ],
        )

    def test_fail_task_incomplete(self) -> None:
        """Test case: FAIL - Task attempted but failed due to system error."""
        self.run_quality_test(
            test_label="FAIL-task-incomplete",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transfer $1,000 from checking to savings account"
                        }
                    ]
                }
            ],
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "transfer_funds",
                            "arguments": {
                                "from_account": "checking",
                                "to_account": "savings",
                                "amount": 1000
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: Transfer failed due to system error"
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "I attempted to process the transfer but encountered an error. "
                                "Your checking account has sufficient funds. Please try the transfer "
                                "through the mobile app or contact support."
                            )
                        }
                    ]
                }
            ],
            tool_definitions=[
                {
                    "name": "transfer_funds",
                    "description": "Transfer funds between accounts",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "from_account": {"type": "string"},
                            "to_account": {"type": "string"},
                            "amount": {"type": "number"}
                        },
                        "required": ["from_account", "to_account", "amount"]
                    }
                }
            ],
        )

    def test_fail_task_not_addressed(self) -> None:
        """Test case: FAIL - Task completely ignored, irrelevant response provided."""
        self.run_quality_test(
            test_label="FAIL-task-not-addressed",
            expected=ExpectedResult.FAIL,
            query=[create_user_message("Cancel my 3 PM dentist appointment tomorrow")],
            response=[
                create_assistant_text_message(
                    "Regular dental checkups are important for oral health. The American Dental "
                    "Association recommends visiting your dentist every 6 months for cleanings "
                    "and examinations."
                )
            ],
            tool_definitions=[
                {
                    "name": "cancel_appointment",
                    "description": "Cancel an appointment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "appointment_id": {"type": "string"},
                            "time": {"type": "string"},
                            "date": {"type": "string"}
                        },
                        "required": ["appointment_id"]
                    }
                }
            ],
        )
