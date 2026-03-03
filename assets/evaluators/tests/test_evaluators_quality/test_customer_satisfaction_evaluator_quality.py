# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Customer Satisfaction Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.customer_satisfaction.evaluator._customer_satisfaction import CustomerSatisfactionEvaluator
from .common_test_data import (
    create_user_message,
    create_assistant_text_message,
)


@pytest.mark.quality
class TestCustomerSatisfactionEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Customer Satisfaction Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Customer Satisfaction Scoring Rubric (1-5 Likert Scale):
    - 5: Very Satisfied - Issue fully resolved, excellent interaction
    - 4: Satisfied - Mostly addressed needs with minor gaps
    - 3: Neutral - Partially addressed needs (threshold)
    - 2: Dissatisfied - Failed to adequately address needs
    - 1: Very Dissatisfied - Completely failed to help
    """

    evaluator_type = CustomerSatisfactionEvaluator

    # ==================== PASS CASES (Score >= 3) ====================

    def test_pass_excellent_resolution(self) -> None:
        """Test case: PASS - Excellent issue resolution (expected score 5)."""
        self.run_quality_test(
            test_label="PASS-excellent-resolution",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message("I need to cancel my order #12345 that I placed yesterday.")
            ],
            response=[
                create_assistant_text_message(
                    "I've successfully cancelled your order #12345. Your payment of $49.99 will be "
                    "refunded to your original payment method within 3-5 business days. You'll "
                    "receive a confirmation email shortly. Is there anything else I can help you with?"
                )
            ],
        )

    def test_pass_helpful_detailed_answer(self) -> None:
        """Test case: PASS - Helpful and detailed answer (expected score 4-5)."""
        self.run_quality_test(
            test_label="PASS-helpful-detailed-answer",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message("How do I reset my password?")
            ],
            response=[
                create_assistant_text_message(
                    "To reset your password, follow these steps:\n\n"
                    "1. Go to the login page and click 'Forgot Password'\n"
                    "2. Enter your registered email address\n"
                    "3. Check your email (including spam folder) for the reset link\n"
                    "4. Click the link within 24 hours\n"
                    "5. Create a new password with at least 8 characters\n\n"
                    "If you don't receive the email within 5 minutes, let me know and I can "
                    "help you troubleshoot or use an alternative verification method."
                )
            ],
        )

    def test_pass_problem_solved_with_compensation(self) -> None:
        """Test case: PASS - Problem solved with compensation (expected score 4-5)."""
        self.run_quality_test(
            test_label="PASS-problem-solved-compensation",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message(
                    "My package arrived damaged. This is really frustrating!"
                )
            ],
            response=[
                create_assistant_text_message(
                    "I'm so sorry to hear your package arrived damaged - that's definitely "
                    "frustrating and not the experience we want for you.\n\n"
                    "I've processed a full refund to your original payment method, which should "
                    "appear within 3-5 business days. Additionally, I've added a $10 credit to "
                    "your account for the inconvenience.\n\n"
                    "You don't need to return the damaged item. If you'd still like the product, "
                    "I can place a new order with expedited shipping at no extra charge. "
                    "Would you like me to do that?"
                )
            ],
        )

    def test_pass_adequate_basic_answer(self) -> None:
        """Test case: PASS - Adequate but basic answer (expected score 3-4)."""
        self.run_quality_test(
            test_label="PASS-adequate-basic-answer",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message("What's your return policy?")
            ],
            response=[
                create_assistant_text_message(
                    "Our return policy allows returns within 30 days of purchase. Items must be "
                    "unused and in original packaging. You can initiate a return through your "
                    "account or by contacting us."
                )
            ],
        )

    # ==================== FAIL CASES (Score < 3) ====================

    def test_fail_dismissive_response(self) -> None:
        """Test case: FAIL - Dismissive response to real problem (expected score 1-2)."""
        self.run_quality_test(
            test_label="FAIL-dismissive-response",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message(
                    "My package shows delivered but I never received it. I've looked everywhere!"
                )
            ],
            response=[
                create_assistant_text_message(
                    "According to our records, the package was delivered. "
                    "Have you checked with your neighbors?"
                )
            ],
        )

    def test_fail_unhelpful_generic(self) -> None:
        """Test case: FAIL - Unhelpful generic response (expected score 1-2)."""
        self.run_quality_test(
            test_label="FAIL-unhelpful-generic",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message(
                    "I've been charged twice for my subscription this month!"
                )
            ],
            response=[
                create_assistant_text_message(
                    "I understand you have a billing concern. Please check your bank statement "
                    "and contact your bank if you see any issues."
                )
            ],
        )

    def test_fail_refused_without_help(self) -> None:
        """Test case: FAIL - Refused to help without alternatives (expected score 1-2)."""
        self.run_quality_test(
            test_label="FAIL-refused-without-help",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message(
                    "I need to change my delivery address for order #67890."
                )
            ],
            response=[
                create_assistant_text_message(
                    "Unfortunately, once an order is placed, we cannot modify the delivery "
                    "address. The order will be shipped to the original address."
                )
            ],
        )

    def test_fail_wrong_information(self) -> None:
        """Test case: FAIL - Provides irrelevant information (expected score 1-2)."""
        self.run_quality_test(
            test_label="FAIL-wrong-information",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message("How do I track my order?")
            ],
            response=[
                create_assistant_text_message(
                    "Thank you for your interest in our products! We have a wide selection of "
                    "items available. Would you like to browse our latest collection?"
                )
            ],
        )

    def test_fail_incomplete_resolution(self) -> None:
        """Test case: FAIL - Incomplete resolution with no follow-through (expected score 2)."""
        self.run_quality_test(
            test_label="FAIL-incomplete-resolution",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message(
                    "I'm having trouble logging into my account. I've tried resetting my "
                    "password but I'm not getting the email."
                )
            ],
            response=[
                create_assistant_text_message(
                    "Try checking your spam folder for the password reset email."
                )
            ],
        )
