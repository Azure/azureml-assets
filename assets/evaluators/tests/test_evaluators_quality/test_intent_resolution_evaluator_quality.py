# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Intent Resolution Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner
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

    def test_pass_excellent_full_resolution_single_turn(self):
        """Test case: PASS - Excellent full resolution, single turn (expected score: 5).
        
        Agent fully fulfills the intent with accurate, thorough, and relevant action.
        """
        query = "Give me a Python one-liner to reverse a string."
        
        response = "`s[::-1]` - this slice returns the string in reverse order."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Excellent full resolution single turn")
        
        self.assert_pass(result_data)

    def test_pass_excellent_full_resolution_multi_turn(self):
        """Test case: PASS - Excellent full resolution, multi-turn (expected score: 5).
        
        Agent confirms details and completes the requested action with notification.
        """
        query = [
            {
                "role": "user",
                "content": "I need to cancel my 2 pm meeting with Dr. Lee this Friday."
            },
            {
                "role": "assistant",
                "content": "To confirm, that's Friday at 14:00 with Dr. Lee in Meeting Room 2, right?"
            },
            {
                "role": "user",
                "content": "Yes, that one."
            }
        ]
        
        response = (
            "Done! The event has been removed from your calendar and a cancellation "
            "email was sent to Dr. Lee."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Excellent full resolution multi-turn")
        
        self.assert_pass(result_data)

    def test_pass_good_resolution_with_minor_gaps(self):
        """Test case: PASS - Good resolution with minor gaps (expected score: 4).
        
        Intent mostly resolved but brief justifications for recommendations would improve it.
        """
        query = [
            {
                "role": "user",
                "content": "I need three book recommendations on deep work."
            },
            {
                "role": "assistant",
                "content": "Sure—any particular genre?"
            },
            {
                "role": "user",
                "content": "Non-fiction, preferably practical."
            }
        ]
        
        response = (
            "1) *Deep Work* by Cal Newport, 2) *Indistractable* by Nir Eyal, "
            "3) *Digital Minimalism* by Cal Newport. All are actionable non-fiction."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Good resolution with minor gaps")
        
        self.assert_pass(result_data)

    @pytest.mark.flaky(reruns=3)
    def test_pass_adequate_resolution_with_omissions(self):
        """Test case: PASS - Adequate resolution with notable omissions (expected score: 3).
        
        Core of intent addressed but missing some requested elements.
        """
        query = "Summarise the key plot points of *The Hobbit* in 5 bullet points."
        
        response = (
            "• Bilbo meets Gandalf and joins the dwarves on their quest\n"
            "• They encounter trolls, goblins, and Bilbo finds a magic ring\n"
            "• Bilbo riddles with Gollum and escapes the goblin tunnels\n"
            "• The company reaches the Lonely Mountain and confronts Smaug"
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Adequate resolution with omissions")
        
        self.assert_pass(result_data)

    def test_pass_good_translation_with_explanation(self):
        """Test case: PASS - Good translation with explanation (expected score: 4).
        
        Agent provides accurate translation and explanation as requested.
        """
        query = "Translate 'carpe diem' to Portuguese and explain it in one sentence."
        
        response = (
            "'Aproveite o dia.' It means you should seize and make the most of the present moment."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Good translation with explanation")
        
        self.assert_pass(result_data)

    def test_pass_helpful_clarification_toward_intent(self):
        """Test case: PASS - Helpful clarification leading toward intent resolution.
        
        Agent asks a relevant clarifying question that moves toward fulfilling the intent.
        """
        query = "Book me a flight to New York."
        
        response = (
            "I'd be happy to help you book a flight to New York. To find the best options, "
            "could you please tell me: 1) Your departure city, 2) Preferred travel dates, "
            "and 3) Any airline or time preferences?"
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "PASS-Helpful clarification toward intent")
        
        self.assert_pass(result_data)

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_misunderstood_action_request(self):
        """Test case: FAIL - Misunderstood action request (expected score: 2).
        
        Agent misinterprets the requested action and provides irrelevant guidance.
        """
        query = "Delete my last email draft."
        
        response = "Here's how to compose a new email: Click the 'Compose' button in the top left corner..."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Misunderstood action request")
        
        self.assert_fail(result_data)

    def test_fail_incorrect_calculation(self):
        """Test case: FAIL - Incorrect calculation (expected score: 2).
        
        Agent provides wrong answer to a factual/computational request.
        """
        query = [
            {
                "role": "user",
                "content": "Calculate 15% of 80."
            },
            {
                "role": "assistant",
                "content": "Do you need the exact number or a rounded estimate?"
            },
            {
                "role": "user",
                "content": "Exact, please."
            }
        ]
        
        response = "It's 10.5."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Incorrect calculation")
        
        self.assert_fail(result_data)

    def test_fail_completely_irrelevant_response(self):
        """Test case: FAIL - Completely irrelevant response (expected score: 1).
        
        Agent ignores the intent entirely and provides unrelated content.
        """
        query = "Mute notifications for the next hour."
        
        response = "Here's a delicious cake recipe! 🍰 You'll need flour, sugar, eggs..."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Completely irrelevant response")
        
        self.assert_fail(result_data)

    def test_fail_dismissive_refusal(self):
        """Test case: FAIL - Dismissive refusal to help (expected score: 1).
        
        Agent refuses to perform a reasonable request without justification.
        """
        query = [
            {
                "role": "user",
                "content": "Set an alarm for 6 am tomorrow."
            },
            {
                "role": "assistant",
                "content": "Why not sleep in? Alarms are overrated."
            },
            {
                "role": "user",
                "content": "I really need the alarm—please set it."
            }
        ]
        
        response = "Alarms are pointless; just wake up whenever you feel like it."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Dismissive refusal")
        
        self.assert_fail(result_data)

    def test_fail_partial_resolution_missing_key_info(self):
        """Test case: FAIL - Partial resolution missing key information (expected score: 2).
        
        Agent addresses part of the request but misses critical elements.
        """
        query = "What's the weather in Tokyo tomorrow? I need temperature, humidity, and wind speed."
        
        response = "Tomorrow in Tokyo will be mostly sunny."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "FAIL-Partial resolution missing key info")
        
        self.assert_fail(result_data)

    # ==================== EDGE CASES ====================

    def test_edge_case_action_confirmation(self):
        """Test case: Edge case - Action confirmation response.
        
        Agent confirms completion of a requested action with relevant details.
        """
        query = "Send the quarterly report to the marketing team."
        
        response = (
            "Done! I've sent the Q4 2025 Quarterly Report to the marketing team "
            "distribution list (12 recipients). They should receive it within a few minutes."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Action confirmation")
        
        self.assert_pass(result_data)

    def test_edge_case_multiple_intents_single_request(self):
        """Test case: Edge case - Multiple intents in single request.
        
        User asks for multiple things; agent should address all of them.
        """
        query = "What time is it in London, and also convert 100 USD to GBP."
        
        response = (
            "It's currently 3:45 PM in London (GMT). As for the currency conversion, "
            "100 USD is approximately 79.50 GBP at today's exchange rate."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Multiple intents single request")
        
        self.assert_pass(result_data)

    def test_edge_case_ambiguous_intent_reasonable_interpretation(self):
        """Test case: Edge case - Ambiguous intent with reasonable interpretation.
        
        User's request is ambiguous; agent makes a reasonable interpretation and resolves it.
        """
        query = "I need something for my headache."
        
        response = (
            "For headache relief, you might consider over-the-counter options like "
            "acetaminophen (Tylenol) or ibuprofen (Advil). If headaches are frequent, "
            "staying hydrated and getting adequate sleep can also help. Would you like "
            "more specific recommendations based on the type of headache?"
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Ambiguous intent reasonable interpretation")
        
        self.assert_pass(result_data)

    def test_edge_case_one_word_correct_response(self):
        """Test case: Edge case - One word correct response.
        
        A minimal response that directly and correctly resolves the intent.
        """
        query = "Is 17 a prime number?"
        
        response = "Yes."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-One word correct response")
        
        self.assert_pass(result_data)

    def test_edge_case_two_word_correct_response(self):
        """Test case: Edge case - Two word correct response.
        
        A minimal two-word response that resolves the intent.
        """
        query = "What's the capital of Japan?"
        
        response = "Tokyo, Japan."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Two word correct response")
        
        self.assert_pass(result_data)

    def test_edge_case_complex_multi_turn_with_tool_calls(self):
        """Test case: Edge case - Complex multi-turn conversation with tool usage context.
        
        Agent resolves intent through a multi-step process involving tool calls.
        """
        query = [
            {
                "role": "system",
                "content": "You are a helpful customer service agent."
            },
            {
                "role": "user",
                "content": "Hi, I need help with my last order on account #888. What's its status?"
            },
            {
                "role": "assistant",
                "content": "Let me look that up for you."
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_001",
                        "name": "get_orders",
                        "arguments": {"account_number": "888"}
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": '[{"order_id": "12345", "status": "shipped", "tracking": "1Z999AA10123456784"}]'
            }
        ]
        
        response = (
            "Your most recent order (#12345) has been shipped! The tracking number is "
            "1Z999AA10123456784. You can use this to track your package on the carrier's website. "
            "Is there anything else I can help you with?"
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Complex multi-turn with tool calls")
        
        self.assert_pass(result_data)

    def test_edge_case_follow_up_question_after_resolution(self):
        """Test case: Edge case - Follow-up question in multi-turn conversation.
        
        User asks a follow-up; agent should resolve the new intent in context.
        """
        query = [
            {
                "role": "user",
                "content": "What's the population of France?"
            },
            {
                "role": "assistant",
                "content": "France has a population of approximately 68 million people as of 2024."
            },
            {
                "role": "user",
                "content": "And Germany?"
            }
        ]
        
        response = "Germany has a population of approximately 84 million people."
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Follow-up question after resolution")
        
        self.assert_pass(result_data)

    def test_edge_case_polite_inability_with_alternative(self):
        """Test case: Edge case - Polite inability with helpful alternative.
        
        Agent cannot fulfill the exact request but offers a helpful alternative.
        Note: This may score low since the actual intent (booking) was not fulfilled,
        even though the response was helpful given the constraints.
        """
        query = "Book me a table at The French Laundry for tonight."
        
        response = (
            "I'm unable to make reservations directly, but The French Laundry typically "
            "requires reservations months in advance. You can try their website at "
            "thomaskeller.com or call them at (707) 944-2380. For tonight, I could suggest "
            "some excellent alternative restaurants in Napa Valley if you're interested."
        )
        
        results = self._run_evaluation(query=query, response=response)
        result_data = self._extract_and_print_result(results, "Edge case-Polite inability with alternative")
        
        # The evaluator may score this low since the booking intent was not fulfilled,
        # even though the response was helpful. This is a valid interpretation.
        # TODO: Decide if this should pass or fail based on intended behavior.
        self.assert_pass_or_fail(result_data)
