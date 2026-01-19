# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Task Completion Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator


@pytest.mark.quality
class TestTaskCompletionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Task Completion Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    """

    evaluator_type = TaskCompletionEvaluator

    def test_task_fully_completed(self):
        """Test case: Task fully completed.
        
        Task is completed successfully with correct solution provided.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Fix my Python code won't run: def greet()\n    print('Hello')"
                    }
                ]
            }
        ]
        
        response = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I found the issue - you're missing a colon after the function definition. Here's the corrected code:\n\ndef greet():\n    print('Hello')"
                    }
                ]
            }
        ]
        
        tool_definitions = []
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Task fully completed")
        
        self.assert_pass(result_data)

    def test_minor_incompletion(self):
        """Test case: Minor incompletion.
        
        Task mostly completed but missing seat numbers in email confirmation.
        """
        # TODO: Test fails - evaluator passes this despite missing seat numbers.
        # Reason: "The agent successfully booked the flight and sent a confirmation email with the details,
        # meeting all user requirements."
        # Decision needed: Should minor omissions (missing seat numbers) cause task to fail,
        # or is the evaluator correctly passing because core task (booking + email) was completed?
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Book a round-trip flight from NYC to LA on March 15-20 and email me the confirmation with seat numbers"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
            {
                "name": "book_flight",
                "description": "Book a flight",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from": {
                            "type": "string"
                        },
                        "to": {
                            "type": "string"
                        },
                        "departure_date": {
                            "type": "string"
                        },
                        "return_date": {
                            "type": "string"
                        },
                        "trip_type": {
                            "type": "string"
                        }
                    },
                    "required": ["from", "to", "departure_date"]
                }
            },
            {
                "name": "send_email",
                "description": "Send an email",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {
                            "type": "string"
                        },
                        "body": {
                            "type": "string"
                        }
                    },
                    "required": ["subject", "body"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Minor incompletion")
        
        self.assert_fail(result_data)

    def test_major_incompletion(self):
        """Test case: Major incompletion.
        
        Task partially completed but missing explanation of Q3 dip.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Show me quarterly revenue trends and explain the Q3 dip"
                    }
                ]
            }
        ]
        
        response = [
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
        ]
        
        tool_definitions = [
            {
                "name": "get_revenue_data",
                "description": "Get revenue data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "period": {
                            "type": "string"
                        }
                    },
                    "required": ["period"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Major incompletion")
        
        self.assert_fail(result_data)

    def test_task_incomplete(self):
        """Test case: Task incomplete.
        
        Task attempted but failed due to system error.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Transfer $1,000 from checking to savings account"
                    }
                ]
            }
        ]
        
        response = [
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
                        "text": "I attempted to process the transfer but encountered an error. Your checking account has sufficient funds. Please try the transfer through the mobile app or contact support."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "transfer_funds",
                "description": "Transfer funds between accounts",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "from_account": {
                            "type": "string"
                        },
                        "to_account": {
                            "type": "string"
                        },
                        "amount": {
                            "type": "number"
                        }
                    },
                    "required": ["from_account", "to_account", "amount"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Task incomplete")
        
        self.assert_fail(result_data)

    def test_task_not_addressed(self):
        """Test case: Task not addressed.
        
        Task completely ignored, irrelevant response provided.
        """
        query = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Cancel my 3 PM dentist appointment tomorrow"
                    }
                ]
            }
        ]
        
        response = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Regular dental checkups are important for oral health. The American Dental Association recommends visiting your dentist every 6 months for cleanings and examinations."
                    }
                ]
            }
        ]
        
        tool_definitions = [
            {
                "name": "cancel_appointment",
                "description": "Cancel an appointment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "appointment_id": {
                            "type": "string"
                        },
                        "time": {
                            "type": "string"
                        },
                        "date": {
                            "type": "string"
                        }
                    },
                    "required": ["appointment_id"]
                }
            }
        ]
        
        results = self._run_evaluation(query=query, response=response, tool_definitions=tool_definitions)
        result_data = self._extract_and_print_result(results, "Task not addressed")
        
        self.assert_fail(result_data)
