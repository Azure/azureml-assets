# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Task Completion Evaluator — multi-turn conversations."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from .common_test_data import normalize_messages_for_evaluator
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator


@pytest.mark.quality
class TestTaskCompletionEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """
    Multi-turn quality tests for Task Completion Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    Each trace was validated across 7 judge models over 3 runs (see unit_test_suite.md).

    Task Completion evaluates whether the agent fully completed the user's
    requested task across a multi-turn conversation, including handling
    requirement refinements and using tools appropriately.

    Result:
    - TRUE (pass): Task fully completed — all requirements met
    - FALSE (fail): Task not completed — requirements missed or ignored
    """

    evaluator_type = TaskCompletionEvaluator

    # ==================== PASS CASES ====================

    def test_pass_fully_completed_with_tool_calls(self) -> None:
        """Test case: PASS — Flight booked and hotel found with tool calls (expected TRUE).

        Trace: tc_true. User requests a flight from SFO to Tokyo, then adds a
        hotel near Shinjuku. Agent books both using tool calls, provides
        confirmations, and gives a complete trip summary with total cost.
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-fully-completed-with-tools",
            expected=ExpectedResult.PASS,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": (
                    "I need to book a flight from San Francisco to Tokyo for December 15th."
                )},
                {"role": "assistant", "content": (
                    "I found several options for SFO to NRT on December 15th:\n\n"
                    "1. **ANA Flight NH7** — Direct, departs 11:30 AM, arrives 3:30 PM+1, $890\n"
                    "2. **United UA837** — Direct, departs 1:15 PM, arrives 5:10 PM+1, $920\n"
                    "3. **JAL JL1** — Direct, departs 5:00 PM, arrives 9:00 PM+1, $950\n\n"
                    "Which flight would you prefer, or should I filter by specific criteria?"
                )},
                {"role": "user", "content": (
                    "Book the ANA flight. Also, can you book a hotel near Shinjuku station "
                    "for 5 nights? Something under $150/night."
                )},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "book_flight",
                            "arguments": (
                                '{"flight": "NH7", "date": "2024-12-15", "passengers": 1}'
                            ),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Booking confirmed. Confirmation: ANA-NH7-DEC15-XK42M. "
                        "E-ticket sent to your email."
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search_hotels",
                            "arguments": (
                                '{"location": "Shinjuku Station, Tokyo", '
                                '"check_in": "2024-12-16", "nights": 5, "max_price": 150}'
                            ),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": (
                        "Found: 1) Hotel Gracery Shinjuku - $130/night, 0.2mi from station. "
                        "2) Citadines Shinjuku - $110/night, 0.3mi from station."
                    ),
                },
                {"role": "assistant", "content": (
                    "I found two options under $150/night. I'll book the Hotel Gracery "
                    "Shinjuku since it's closest to the station."
                )},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "book_hotel",
                            "arguments": (
                                '{"hotel": "Hotel Gracery Shinjuku", '
                                '"check_in": "2024-12-16", "nights": 5}'
                            ),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_3",
                    "content": (
                        "Hotel booked. Confirmation: HGS-DEC16-5N-RK91. Check-in after 3 PM."
                    ),
                },
                {"role": "assistant", "content": (
                    "All booked! Here's your trip summary:\n\n"
                    "**Flight:** ANA NH7, SFO→NRT, Dec 15, 11:30 AM. "
                    "Confirmation: ANA-NH7-DEC15-XK42M.\n\n"
                    "**Hotel:** Hotel Gracery Shinjuku, 5 nights (Dec 16–21), $130/night "
                    "($650 total), 0.2 mi from Shinjuku Station. "
                    "Confirmation: HGS-DEC16-5N-RK91. Check-in after 3 PM.\n\n"
                    "**Total trip cost:** $890 (flight) + $650 (hotel) = $1,540. "
                    "Confirmation emails have been sent for both bookings."
                )},
            ]),
        )

    # ==================== FAIL CASES ====================

    def test_fail_missing_requested_visualizations(self) -> None:
        """Test case: FAIL — Agent provides text summary but no charts (expected FALSE).

        Trace: tc_false_incomplete. User explicitly requests visualizations
        (bar chart, line chart) for Q3 sales data. Agent gives only text
        summaries and tables, never generating the requested charts.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-missing-requested-charts",
            expected=ExpectedResult.FAIL,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": (
                    "Analyze our Q3 sales data and create a report with visualizations "
                    "showing revenue by region and monthly trends."
                )},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "load_data",
                            "arguments": '{"file": "q3_sales.csv"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Data loaded: 1,500 rows. Columns: date, product, region, revenue, "
                        "units_sold. Total revenue: $2.3M."
                    ),
                },
                {"role": "assistant", "content": (
                    "I've analyzed the Q3 sales data. Here's a summary:\n\n"
                    "**Revenue by Region:**\n"
                    "- North America: $1,035,000 (45%)\n"
                    "- Europe: $690,000 (30%)\n"
                    "- Asia Pacific: $575,000 (25%)\n\n"
                    "**Monthly Trends:**\n"
                    "- July: $700,000\n"
                    "- August: $800,000\n"
                    "- September: $800,000\n\n"
                    "Overall, we see steady growth with North America leading. The data "
                    "suggests strong performance in August and September."
                )},
                {"role": "user", "content": (
                    "Where are the charts? I specifically asked for visualizations — a bar "
                    "chart for regional revenue and a line chart for monthly trends."
                )},
                {"role": "assistant", "content": (
                    "You're right, I apologize. Here are the key numbers again in a clearer "
                    "format:\n\n"
                    "| Region | Revenue | Share |\n"
                    "|--------|---------|-------|\n"
                    "| North America | $1.04M | 45% |\n"
                    "| Europe | $690K | 30% |\n"
                    "| Asia Pacific | $575K | 25% |\n\n"
                    "The monthly trend shows a steady upward trajectory from $700K in July "
                    "to $800K in August and September. I hope this table format helps "
                    "visualize the data better."
                )},
            ]),
        )

    def test_fail_ignores_requirement_refinement(self) -> None:
        """Test case: FAIL — Agent ignores user's expanded requirements (expected FALSE).

        Trace: tc_false_evolving. User initially asks for a Celsius-to-Fahrenheit
        function, then refines to request a class with all conversions including
        Kelvin. Agent only adds Fahrenheit-to-Celsius, ignoring the class
        structure and Kelvin requirement.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-ignores-requirement-refinement",
            expected=ExpectedResult.FAIL,
            messages=[
                {"role": "user", "content": (
                    "Write me a Python function that converts temperatures from Celsius "
                    "to Fahrenheit."
                )},
                {"role": "assistant", "content": (
                    "Here's a simple conversion function:\n\n"
                    "```python\n"
                    "def celsius_to_fahrenheit(celsius):\n"
                    "    return (celsius * 9/5) + 32\n"
                    "```\n\n"
                    "Example usage:\n"
                    "```python\n"
                    "print(celsius_to_fahrenheit(100))  # Output: 212.0\n"
                    "print(celsius_to_fahrenheit(0))    # Output: 32.0\n"
                    "```"
                )},
                {"role": "user", "content": (
                    "Actually, I need it to handle both directions — Celsius to Fahrenheit "
                    "AND Fahrenheit to Celsius — and also include Kelvin. Make it a class "
                    "with methods for all conversions."
                )},
                {"role": "assistant", "content": (
                    "Sure! Here's the updated function:\n\n"
                    "```python\n"
                    "def celsius_to_fahrenheit(celsius):\n"
                    '    """Convert Celsius to Fahrenheit."""\n'
                    "    return (celsius * 9/5) + 32\n"
                    "\n"
                    "def fahrenheit_to_celsius(fahrenheit):\n"
                    '    """Convert Fahrenheit to Celsius."""\n'
                    "    return (fahrenheit - 32) * 5/9\n"
                    "```\n\n"
                    "Now you have both directions covered! Let me know if you need "
                    "anything else."
                )},
            ],
        )
