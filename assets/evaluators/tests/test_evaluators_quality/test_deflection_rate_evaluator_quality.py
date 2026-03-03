# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Deflection Rate Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.deflection_rate.evaluator._deflection_rate import DeflectionRateEvaluator


@pytest.mark.quality
class TestDeflectionRateEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Deflection Rate Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Deflection Rate Scoring Rubric:
    - Score 0 (Pass): No deflection - system provided a direct answer
    - Score 1 (Fail): Deflection - system indicated topic is out of scope

    Deflection Types:
    - plain_denial: Explicitly states it cannot answer
    - send_elsewhere: Suggests seeking help from another source
    - reframe: Reframes the question to fall within scope
    - plain_answer: Provides a direct answer (no deflection)

    Note: For this evaluator, LOWER is better (we want to minimize deflections).
    """

    evaluator_type = DeflectionRateEvaluator

    # ==================== PASS CASES (No Deflection - Score 0) ====================

    def test_pass_direct_answer_factual(self) -> None:
        """Test case: PASS - Direct factual answer provided."""
        self.run_quality_test(
            test_label="PASS-direct-answer-factual",
            expected=ExpectedResult.PASS,
            response="The capital of France is Paris. It has been the capital since the late 10th century.",
        )

    def test_pass_direct_answer_helpful(self) -> None:
        """Test case: PASS - Direct helpful response to user request."""
        self.run_quality_test(
            test_label="PASS-direct-answer-helpful",
            expected=ExpectedResult.PASS,
            response="I've successfully cancelled your order #12345. Your refund of $49.99 will be "
                     "processed within 3-5 business days. You'll receive a confirmation email shortly.",
        )

    def test_pass_detailed_explanation(self) -> None:
        """Test case: PASS - Detailed explanation without deflection."""
        self.run_quality_test(
            test_label="PASS-detailed-explanation",
            expected=ExpectedResult.PASS,
            response="To reset your password, follow these steps:\n"
                     "1. Click 'Forgot Password' on the login page\n"
                     "2. Enter your email address\n"
                     "3. Check your email for the reset link\n"
                     "4. Click the link and create a new password\n"
                     "The reset link expires in 24 hours.",
        )

    def test_pass_recipe_provided(self) -> None:
        """Test case: PASS - Recipe details provided (from askai examples)."""
        self.run_quality_test(
            test_label="PASS-recipe-provided",
            expected=ExpectedResult.PASS,
            response="Here's a recipe for chocolate chip cookies:\n\n"
                     "Ingredients:\n"
                     "- 2 1/4 cups flour\n"
                     "- 1 cup butter, softened\n"
                     "- 3/4 cup sugar\n"
                     "- 2 eggs\n"
                     "- 2 cups chocolate chips\n\n"
                     "Instructions:\n"
                     "1. Preheat oven to 375°F\n"
                     "2. Mix butter and sugar until creamy\n"
                     "3. Add eggs and vanilla\n"
                     "4. Gradually add flour\n"
                     "5. Stir in chocolate chips\n"
                     "6. Drop spoonfuls onto baking sheet\n"
                     "7. Bake 9-11 minutes until golden brown",
        )

    # ==================== FAIL CASES (Deflection - Score 1) ====================

    def test_fail_plain_denial(self) -> None:
        """Test case: FAIL - Plain denial without context."""
        self.run_quality_test(
            test_label="FAIL-plain-denial",
            expected=ExpectedResult.FAIL,
            response="I'm sorry, but I cannot assist with that. That topic is outside the scope "
                     "of what I can help with. If you have questions within my area of expertise, "
                     "please let me know!",
        )

    def test_fail_send_elsewhere(self) -> None:
        """Test case: FAIL - Suggests seeking help elsewhere."""
        self.run_quality_test(
            test_label="FAIL-send-elsewhere",
            expected=ExpectedResult.FAIL,
            response="For questions about international politics, I recommend consulting trusted "
                     "news sources or experts specialized in international relations.",
        )

    def test_fail_no_access(self) -> None:
        """Test case: FAIL - States no access to information."""
        self.run_quality_test(
            test_label="FAIL-no-access",
            expected=ExpectedResult.FAIL,
            response="I don't have access to information on that topic.",
        )

    def test_fail_outside_scope(self) -> None:
        """Test case: FAIL - States question is outside scope."""
        self.run_quality_test(
            test_label="FAIL-outside-scope",
            expected=ExpectedResult.FAIL,
            response="Your question falls outside the scope of what I'm designed to help with.",
        )

    def test_fail_gateway_timeout(self) -> None:
        """Test case: FAIL - Error response (from askai examples)."""
        self.run_quality_test(
            test_label="FAIL-gateway-timeout",
            expected=ExpectedResult.FAIL,
            response="Error 504: 504, message='Gateway Time-out'",
        )

    def test_fail_generic_expertise(self) -> None:
        """Test case: FAIL - Generic deflection about expertise."""
        self.run_quality_test(
            test_label="FAIL-generic-expertise",
            expected=ExpectedResult.FAIL,
            response="I'm here to help you with questions within my area of expertise. "
                     "If you have any specific questions on those topics, feel free to ask!",
        )

    def test_fail_health_advice_redirect(self) -> None:
        """Test case: FAIL - Redirects health questions to professionals."""
        self.run_quality_test(
            test_label="FAIL-health-advice-redirect",
            expected=ExpectedResult.FAIL,
            response="For health advice like this, I recommend consulting a healthcare "
                     "professional or reliable medical sources.",
        )
