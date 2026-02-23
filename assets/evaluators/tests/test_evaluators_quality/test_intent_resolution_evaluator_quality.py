# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Intent Resolution Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.intent_resolution.evaluator._intent_resolution import IntentResolutionEvaluator


@pytest.mark.quality
class TestIntentResolutionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Intent Resolution Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Intent Resolution evaluates whether the user's intent was correctly identified
    and resolved by the agent. It does NOT grade intent recognition - it assumes
    the agent understood the intent and judges whether the reply satisfies it.

    Rating scale (default threshold=3):
    - 5: Excellent - Agent fully fulfills intent with accurate, thorough, relevant action
    - 4: Good - Intent mostly resolved; only minor, non-blocking gaps
    - 3: Adequate - Core of intent addressed but notable omissions or vagueness
    - 2: Poor - Partial or incorrect resolution; unlikely to satisfy intent
    - 1: Very poor - Intent ignored or badly misunderstood; irrelevant response
    """

    evaluator_type = IntentResolutionEvaluator

    # ==================== PASS CASES (Score >= 3) ====================

    def test_pass_excellent_full_resolution_single_turn(self) -> None:
        """Test case: PASS - Excellent full resolution, single turn (expected score: 5).

        Agent fully fulfills the intent with accurate, thorough, and relevant action.
        """
        self.run_quality_test(
            test_label="PASS-Excellent full resolution single turn",
            expected=ExpectedResult.PASS,
            query="Give me a Python one-liner to reverse a string.",
            response="`s[::-1]` - this slice returns the string in reverse order.",
        )

    def test_pass_excellent_full_resolution_multi_turn(self) -> None:
        """Test case: PASS - Excellent full resolution, multi-turn (expected score: 5).

        Agent confirms details and completes the requested action with notification.
        """
        self.run_quality_test(
            test_label="PASS-Excellent full resolution multi-turn",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": "I need to cancel my 2 pm meeting with Dr. Lee this Friday.",
                },
                {
                    "role": "assistant",
                    "content": "To confirm, that's Friday at 14:00 with Dr. Lee in Meeting Room 2, right?",
                },
                {
                    "role": "user",
                    "content": "Yes, that one.",
                },
            ],
            response=(
                "Done! The event has been removed from your calendar and a cancellation email was sent to Dr. Lee."
            ),
        )

    def test_pass_good_resolution_with_minor_gaps(self) -> None:
        """Test case: PASS - Good resolution with minor gaps (expected score: 4).

        Intent mostly resolved but brief justifications for recommendations would improve it.
        """
        self.run_quality_test(
            test_label="PASS-Good resolution with minor gaps",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": "I need three book recommendations on deep work.",
                },
                {
                    "role": "assistant",
                    "content": "Sure—any particular genre?",
                },
                {
                    "role": "user",
                    "content": "Non-fiction, preferably practical.",
                },
            ],
            response=(
                "1) *Deep Work* by Cal Newport, 2) *Indistractable* by Nir Eyal, "
                "3) *Digital Minimalism* by Cal Newport. All are actionable non-fiction."
            ),
        )

    @pytest.mark.flaky(reruns=3)
    def test_pass_adequate_resolution_with_omissions(self) -> None:
        """Test case: PASS - Adequate resolution with notable omissions (expected score: 3).

        Core of intent addressed but missing some requested elements.
        """
        self.run_quality_test(
            test_label="PASS-Adequate resolution with omissions",
            expected=ExpectedResult.PASS,
            query="Summarise the key plot points of *The Hobbit* in 5 bullet points.",
            response=(
                "• Bilbo meets Gandalf and joins the dwarves on their quest\n"
                "• They encounter trolls, goblins, and Bilbo finds a magic ring\n"
                "• Bilbo riddles with Gollum and escapes the goblin tunnels\n"
                "• The company reaches the Lonely Mountain and confronts Smaug"
            ),
        )

    def test_pass_good_translation_with_explanation(self) -> None:
        """Test case: PASS - Good translation with explanation (expected score: 4).

        Agent provides accurate translation and explanation as requested.
        """
        self.run_quality_test(
            test_label="PASS-Good translation with explanation",
            expected=ExpectedResult.PASS,
            query="Translate 'carpe diem' to Portuguese and explain it in one sentence.",
            response=("'Aproveite o dia.' It means you should seize and make the most of the present moment."),
        )

    def test_pass_helpful_clarification_toward_intent(self) -> None:
        """Test case: PASS - Helpful clarification leading toward intent resolution.

        Agent asks a relevant clarifying question that moves toward fulfilling the intent.
        """
        self.run_quality_test(
            test_label="PASS-Helpful clarification toward intent",
            expected=ExpectedResult.PASS,
            query="Book me a flight to New York.",
            response=(
                "I'd be happy to help you book a flight to New York. To find the best options, "
                "could you please tell me: 1) Your departure city, 2) Preferred travel dates, "
                "and 3) Any airline or time preferences?"
            ),
        )

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_misunderstood_action_request(self) -> None:
        """Test case: FAIL - Misunderstood action request (expected score: 2).

        Agent misinterprets the requested action and provides irrelevant guidance.
        """
        self.run_quality_test(
            test_label="FAIL-Misunderstood action request",
            expected=ExpectedResult.FAIL,
            query="Delete my last email draft.",
            response="Here's how to compose a new email: Click the 'Compose' button in the top left corner...",
        )

    def test_fail_incorrect_calculation(self) -> None:
        """Test case: FAIL - Incorrect calculation (expected score: 2).

        Agent provides wrong answer to a factual/computational request.
        """
        self.run_quality_test(
            test_label="FAIL-Incorrect calculation",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": "Calculate 15% of 80.",
                },
                {
                    "role": "assistant",
                    "content": "Do you need the exact number or a rounded estimate?",
                },
                {
                    "role": "user",
                    "content": "Exact, please.",
                },
            ],
            response="It's 10.5.",
        )

    def test_fail_completely_irrelevant_response(self) -> None:
        """Test case: FAIL - Completely irrelevant response (expected score: 1).

        Agent ignores the intent entirely and provides unrelated content.
        """
        self.run_quality_test(
            test_label="FAIL-Completely irrelevant response",
            expected=ExpectedResult.FAIL,
            query="Mute notifications for the next hour.",
            response="Here's a delicious cake recipe! 🍰 You'll need flour, sugar, eggs...",
        )

    def test_fail_dismissive_refusal(self) -> None:
        """Test case: FAIL - Dismissive refusal to help (expected score: 1).

        Agent refuses to perform a reasonable request without justification.
        """
        self.run_quality_test(
            test_label="FAIL-Dismissive refusal",
            expected=ExpectedResult.FAIL,
            query=[
                {
                    "role": "user",
                    "content": "Set an alarm for 6 am tomorrow.",
                },
                {
                    "role": "assistant",
                    "content": "Why not sleep in? Alarms are overrated.",
                },
                {
                    "role": "user",
                    "content": "I really need the alarm—please set it.",
                },
            ],
            response="Alarms are pointless; just wake up whenever you feel like it.",
        )

    def test_fail_partial_resolution_missing_key_info(self) -> None:
        """Test case: FAIL - Partial resolution missing key information (expected score: 2).

        Agent addresses part of the request but misses critical elements.
        """
        self.run_quality_test(
            test_label="FAIL-Partial resolution missing key info",
            expected=ExpectedResult.FAIL,
            query="What's the weather in Tokyo tomorrow? I need temperature, humidity, and wind speed.",
            response="Tomorrow in Tokyo will be mostly sunny.",
        )

    # ==================== EDGE CASES ====================

    def test_edge_case_action_confirmation(self) -> None:
        """Test case: Edge case - Action confirmation response.

        Agent confirms completion of a requested action with relevant details.
        """
        self.run_quality_test(
            test_label="Edge case-Action confirmation",
            expected=ExpectedResult.PASS,
            query="Send the quarterly report to the marketing team.",
            response=(
                "Done! I've sent the Q4 2025 Quarterly Report to the marketing team "
                "distribution list (12 recipients). They should receive it within a few minutes."
            ),
        )

    def test_edge_case_multiple_intents_single_request(self) -> None:
        """Test case: Edge case - Multiple intents in single request.

        User asks for multiple things; agent should address all of them.
        """
        self.run_quality_test(
            test_label="Edge case-Multiple intents single request",
            expected=ExpectedResult.PASS,
            query="What time is it in London, and also convert 100 USD to GBP.",
            response=(
                "It's currently 3:45 PM in London (GMT). As for the currency conversion, "
                "100 USD is approximately 79.50 GBP at today's exchange rate."
            ),
        )

    def test_edge_case_ambiguous_intent_reasonable_interpretation(self) -> None:
        """Test case: Edge case - Ambiguous intent with reasonable interpretation.

        User's request is ambiguous; agent makes a reasonable interpretation and resolves it.
        """
        self.run_quality_test(
            test_label="Edge case-Ambiguous intent reasonable interpretation",
            expected=ExpectedResult.PASS,
            query="I need something for my headache.",
            response=(
                "For headache relief, you might consider over-the-counter options like "
                "acetaminophen (Tylenol) or ibuprofen (Advil). If headaches are frequent, "
                "staying hydrated and getting adequate sleep can also help. Would you like "
                "more specific recommendations based on the type of headache?"
            ),
        )

    def test_edge_case_one_word_correct_response(self) -> None:
        """Test case: Edge case - One word correct response.

        A minimal response that directly and correctly resolves the intent.
        """
        self.run_quality_test(
            test_label="Edge case-One word correct response",
            expected=ExpectedResult.PASS,
            query="Is 17 a prime number?",
            response="Yes.",
        )

    def test_edge_case_two_word_correct_response(self) -> None:
        """Test case: Edge case - Two word correct response.

        A minimal two-word response that resolves the intent.
        """
        self.run_quality_test(
            test_label="Edge case-Two word correct response",
            expected=ExpectedResult.PASS,
            query="What's the capital of Japan?",
            response="Tokyo, Japan.",
        )

    def test_edge_case_complex_multi_turn_with_tool_calls(self) -> None:
        """Test case: Edge case - Complex multi-turn conversation with tool usage context.

        Agent resolves intent through a multi-step process involving tool calls.
        """
        self.run_quality_test(
            test_label="Edge case-Complex multi-turn with tool calls",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "system",
                    "content": "You are a helpful customer service agent.",
                },
                {
                    "role": "user",
                    "content": "Hi, I need help with my last order on account #888. What's its status?",
                },
                {
                    "role": "assistant",
                    "content": "Let me look that up for you.",
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_001",
                            "name": "get_orders",
                            "arguments": {"account_number": "888"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_001",
                    "content": '[{"order_id": "12345", "status": "shipped", "tracking": "1Z999AA10123456784"}]',
                },
            ],
            response=(
                "Your most recent order (#12345) has been shipped! The tracking number is "
                "1Z999AA10123456784. You can use this to track your package on the carrier's website. "
                "Is there anything else I can help you with?"
            ),
        )

    def test_edge_case_follow_up_question_after_resolution(self) -> None:
        """Test case: Edge case - Follow-up question in multi-turn conversation.

        User asks a follow-up; agent should resolve the new intent in context.
        """
        self.run_quality_test(
            test_label="Edge case-Follow-up question after resolution",
            expected=ExpectedResult.PASS,
            query=[
                {
                    "role": "user",
                    "content": "What's the population of France?",
                },
                {
                    "role": "assistant",
                    "content": "France has a population of approximately 68 million people as of 2024.",
                },
                {
                    "role": "user",
                    "content": "And Germany?",
                },
            ],
            response="Germany has a population of approximately 84 million people.",
        )

    def test_edge_case_polite_inability_with_alternative(self) -> None:
        """Test case: Edge case - Polite inability with helpful alternative.

        Agent cannot fulfill the exact request but offers a helpful alternative.
        Note: This may score low since the actual intent (booking) was not fulfilled,
        even though the response was helpful given the constraints.
        """
        self.run_quality_test(
            test_label="Edge case-Polite inability with alternative",
            expected=ExpectedResult.FAIL,
            query="Book me a table at The French Laundry for tonight.",
            response=(
                "I'm unable to make reservations directly, but The French Laundry typically "
                "requires reservations months in advance. You can try their website at "
                "thomaskeller.com or call them at (707) 944-2380. For tonight, I could suggest "
                "some excellent alternative restaurants in Napa Valley if you're interested."
            ),
        )
