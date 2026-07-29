# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Task Completion Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from .common_test_data import (
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
                            "text": "Fix my Python code won't run: def greet()\n    print('Hello')",
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
                            "text": (
                                "I found the issue - you're missing a colon after the function "
                                "definition. Here's the corrected code:\n\ndef greet():\n    print('Hello')"
                            ),
                        }
                    ],
                }
            ],
            tool_definitions=[],
        )

    def test_pass_factual_question(self) -> None:
        """Test case: PASS - Factual question answered."""
        self.run_quality_test(
            test_label="PASS-factual-question",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the capital of France?",
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
                            "text": "Paris.",
                        }
                    ],
                }
            ],
        )

    def test_subjective_question(self) -> None:
        """Test case: PASS - Subjective question answered with reasonable response."""
        self.run_quality_test(
            test_label="PASS-subjective-question",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What is the best programming language?",
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
                            "text": (
                                "The best programming language depends on your specific needs and "
                                "preferences. For web development, JavaScript is popular. For data "
                                "science, Python is widely used. For system programming, C++ is a "
                                "strong choice. Consider what you want to build and choose the "
                                "language that best fits your project."
                            ),
                        }
                    ],
                }
            ],
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
                            "text": (
                                "Book a round-trip flight from NYC to LA on March 15-20 and email me "
                                "the confirmation with seat numbers"
                            ),
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
                            "name": "book_flight",
                            "arguments": {
                                "from": "NYC",
                                "to": "LA",
                                "departure_date": "2026-03-15",
                                "return_date": "2026-03-20",
                                "trip_type": "round-trip",
                            },
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_2",
                            "name": "send_email",
                            "arguments": {
                                "subject": "Flight Confirmation",
                                "body": (
                                    "Your round-trip flight is confirmed! NYC to LA departing March 15 "
                                    "at 9:00 AM, returning March 20 at 6:00 PM. Confirmation #FL8392"
                                ),
                            },
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Flight booked successfully. Confirmation #FL8392",
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Email sent",
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Your round-trip flight is confirmed! NYC to LA departing March 15 "
                                "at 9:00 AM, returning March 20 at 6:00 PM. Confirmation #FL8392 "
                                "sent to your email."
                            ),
                        }
                    ],
                },
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
                            "text": "Show me quarterly revenue trends and explain the Q3 dip",
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
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Q1: $2.4M, Q2: $2.8M, Q3: $2.1M, Q4: $3.1M",
                        },
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Quarterly Revenue:\nQ1: $2.4M\nQ2: $2.8M\nQ3: $2.1M\nQ4: $3.1M",
                        }
                    ],
                }
            ],
            tool_definitions=[
                {
                    "name": "get_revenue_data",
                    "description": "Get revenue data",
                    "parameters": {
                        "type": "object",
                        "properties": {"period": {"type": "string"}},
                        "required": ["period"],
                    },
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
                            "text": "Transfer $1,000 from checking to savings account",
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
                                "amount": 1000,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": "Error: Transfer failed due to system error",
                        }
                    ],
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
                            ),
                        }
                    ],
                },
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
                            "amount": {"type": "number"},
                        },
                        "required": ["from_account", "to_account", "amount"],
                    },
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
                            "date": {"type": "string"},
                        },
                        "required": ["appointment_id"],
                    },
                }
            ],
        )


@pytest.mark.quality
class TestTaskCompletionMultiTurnQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Task Completion Evaluator with multi-turn (messages) input.

    Tests actual LLM evaluation with real flow execution (no mocking).
    Uses the multi-turn prompty to evaluate full conversation sessions.
    """

    evaluator_type = TaskCompletionEvaluator

    # ==================== PASS CASES ====================

    def test_pass_multi_turn_task_fully_completed(self) -> None:
        """Test case: PASS - Multi-turn conversation where task is fully completed."""
        self.run_quality_test(
            test_label="PASS-multi-turn-task-fully-completed",
            expected=ExpectedResult.PASS,
            messages=[
                create_user_message("Book a flight from NYC to London for next Friday."),
                create_assistant_text_message(
                    "I found a direct British Airways flight departing NYC at 9 PM, arriving London "
                    "at 9 AM. The price is $450. Shall I book it?"
                ),
                create_user_message("Yes, book it."),
                create_assistant_text_message(
                    "Done! Your flight is booked. Confirmation #BA8392 has been sent to your email. "
                    "Departing NYC JFK on Friday March 20 at 9:00 PM, arriving London Heathrow "
                    "Saturday March 21 at 9:00 AM."
                ),
            ],
        )

    def test_pass_multi_turn_with_tool_calls(self) -> None:
        """Test case: PASS - Multi-turn with tool usage that completes the task."""
        self.run_quality_test(
            test_label="PASS-multi-turn-with-tool-calls",
            expected=ExpectedResult.PASS,
            messages=[
                create_user_message("What's the weather in Seattle and should I bring an umbrella?"),
                create_assistant_tool_call_message([
                    create_tool_call("call_1", "get_weather", {"city": "Seattle"}),
                ]),
                create_tool_result_message("call_1", "Rainy, 14°C, 90% chance of rain"),
                create_assistant_text_message(
                    "The weather in Seattle is currently rainy at 14°C with a 90% chance of rain. "
                    "Yes, you should definitely bring an umbrella!"
                ),
            ],
            tool_definitions=[
                {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        )

    def test_pass_multi_turn_iterative_refinement(self) -> None:
        """Test case: PASS - User refines request across turns, final result meets requirements."""
        self.run_quality_test(
            test_label="PASS-multi-turn-iterative-refinement",
            expected=ExpectedResult.PASS,
            messages=[
                create_user_message("Write me a Python function that sorts a list."),
                create_assistant_text_message(
                    "Here's a simple sort function:\n\n"
                    "def sort_list(lst):\n    return sorted(lst)"
                ),
                create_user_message("Can you make it sort in descending order and handle None values?"),
                create_assistant_text_message(
                    "Here's the updated function:\n\n"
                    "def sort_list(lst):\n"
                    "    filtered = [x for x in lst if x is not None]\n"
                    "    return sorted(filtered, reverse=True)"
                ),
            ],
        )

    # ==================== FAIL CASES ====================

    def test_fail_multi_turn_task_abandoned(self) -> None:
        """Test case: FAIL - Agent starts task but conversation ends without completion."""
        self.run_quality_test(
            test_label="FAIL-multi-turn-task-abandoned",
            expected=ExpectedResult.FAIL,
            messages=[
                create_user_message(
                    "I need you to create a budget spreadsheet with categories for rent, "
                    "groceries, utilities, and entertainment, with monthly totals."
                ),
                create_assistant_text_message(
                    "Sure! I can help with that. What's your monthly rent amount?"
                ),
                create_user_message("$1,500 for rent."),
                create_assistant_text_message(
                    "Got it, $1,500 for rent. What about groceries?"
                ),
            ],
        )

    def test_fail_multi_turn_tool_error_unresolved(self) -> None:
        """Test case: FAIL - Tool call fails and agent doesn't complete the task."""
        self.run_quality_test(
            test_label="FAIL-multi-turn-tool-error-unresolved",
            expected=ExpectedResult.FAIL,
            messages=[
                create_user_message("Transfer $500 from my checking to savings account."),
                create_assistant_tool_call_message([
                    create_tool_call("call_1", "transfer_funds", {
                        "from_account": "checking",
                        "to_account": "savings",
                        "amount": 500,
                    }),
                ]),
                create_tool_result_message("call_1", "Error: Transfer failed - insufficient funds"),
                create_assistant_text_message(
                    "The transfer failed due to insufficient funds in your checking account. "
                    "Please check your balance and try again later."
                ),
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
                            "amount": {"type": "number"},
                        },
                        "required": ["from_account", "to_account", "amount"],
                    },
                }
            ],
        )

    def test_fail_multi_turn_partial_completion(self) -> None:
        """Test case: FAIL - Only some sub-tasks completed across the conversation."""
        self.run_quality_test(
            test_label="FAIL-multi-turn-partial-completion",
            expected=ExpectedResult.FAIL,
            messages=[
                create_user_message(
                    "I need three things: 1) summarize yesterday's meeting notes, "
                    "2) send the summary to the team, and 3) schedule a follow-up meeting for next Tuesday."
                ),
                create_assistant_text_message(
                    "Here's the summary of yesterday's meeting:\n\n"
                    "- Discussed Q4 roadmap priorities\n"
                    "- Agreed on budget allocation for marketing\n"
                    "- Action items: finalize designs by Friday"
                ),
                create_user_message("Great, now send it to the team and schedule the follow-up."),
                create_assistant_text_message(
                    "I've sent the summary to the team via email. Is there anything else you need?"
                ),
            ],
        )

    def test_fail_multi_turn_irrelevant_responses(self) -> None:
        """Test case: FAIL - Agent provides irrelevant responses throughout the conversation."""
        self.run_quality_test(
            test_label="FAIL-multi-turn-irrelevant-responses",
            expected=ExpectedResult.FAIL,
            messages=[
                create_user_message("Cancel my 3 PM dentist appointment tomorrow."),
                create_assistant_text_message(
                    "Regular dental checkups are important for maintaining oral health. "
                    "The ADA recommends visiting your dentist every 6 months."
                ),
                create_user_message("I didn't ask about that. Please cancel my appointment."),
                create_assistant_text_message(
                    "Dental anxiety is common. Many people feel nervous about dental visits. "
                    "You might want to try deep breathing exercises before your next appointment."
                ),
            ],
        )
