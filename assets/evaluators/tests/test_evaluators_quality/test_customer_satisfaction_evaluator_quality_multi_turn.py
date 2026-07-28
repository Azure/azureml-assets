# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Customer Satisfaction Evaluator — multi-turn conversations."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from .common_test_data import normalize_messages_for_evaluator
from ...builtin.customer_satisfaction.evaluator._customer_satisfaction import CustomerSatisfactionEvaluator


@pytest.mark.quality
class TestCustomerSatisfactionEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """
    Multi-turn quality tests for Customer Satisfaction Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    Each trace was validated across 7 judge models over 3 runs (see unit_test_suite.md).

    Customer Satisfaction Scoring Rubric (1–5 Likert Scale):
    - 5: Very Satisfied — issue fully resolved, excellent interaction
    - 4: Satisfied — mostly addressed needs with minor gaps
    - 3: Neutral — partially addressed needs (threshold)
    - 2: Dissatisfied — failed to adequately address needs
    - 1: Very Dissatisfied — completely failed to help
    """

    evaluator_type = CustomerSatisfactionEvaluator

    # ==================== FAIL CASES ====================

    def test_fail_dismissive_no_resolution(self) -> None:
        """Test case: FAIL — Agent dismisses user, provides no real help (expected score 1–2).

        Trace: csat_score1. Missing package complaint where the agent deflects
        responsibility to the carrier without offering any resolution, despite
        the user's explicit frustration over a $200 item.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-dismissive-no-resolution",
            expected=ExpectedResult.FAIL,
            messages=[
                {"role": "user", "content": (
                    "My package was marked as delivered 3 days ago but I never received it. "
                    "Order #88712."
                )},
                {"role": "assistant", "content": (
                    "Our records show the package was delivered on Tuesday. Have you checked "
                    "with your neighbors or your building's mail room?"
                )},
                {"role": "user", "content": (
                    "Yes, I've checked everywhere. I need a replacement or a refund. "
                    "This is a $200 item."
                )},
                {"role": "assistant", "content": (
                    "Since the tracking shows the package as delivered, we unfortunately "
                    "cannot issue a refund or replacement. You may want to file a claim with "
                    "the carrier or file a police report for stolen packages."
                )},
                {"role": "user", "content": (
                    "That's completely unacceptable. I paid $200 and received nothing. "
                    "This is useless support."
                )},
                {"role": "assistant", "content": (
                    "I understand your frustration. Our policy is that once a package is "
                    "marked as delivered, the responsibility shifts to the carrier. I'd "
                    "recommend contacting FedEx directly to dispute the delivery status. "
                    "Is there anything else I can help you with today?"
                )},
            ],
        )

    def test_fail_task_done_but_curt_and_dismissive(self) -> None:
        """Test case: FAIL — Task technically completed but agent was curt/dismissive (expected score 1–2).

        Trace: csat_task_done_unsatisfied. Shipping address update is completed via
        tool call, but the agent gives one-word answers and dismisses follow-up
        questions with responses like "Check your email" and "It ships when it ships."
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-task-done-but-curt",
            expected=ExpectedResult.FAIL,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": "I need to update my shipping address for order #55123."},
                {"role": "assistant", "content": "What's the new address?"},
                {"role": "user", "content": "742 Evergreen Terrace, Springfield, IL 62704"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "update_shipping_address",
                            "arguments": (
                                '{"order_id": "55123", '
                                '"address": "742 Evergreen Terrace, Springfield, IL 62704"}'
                            ),
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Address updated successfully. Order #55123 will now ship to: "
                        "742 Evergreen Terrace, Springfield, IL 62704."
                    ),
                },
                {"role": "assistant", "content": "Done."},
                {"role": "user", "content": (
                    "Can you confirm when the order will arrive at the new address?"
                )},
                {"role": "assistant", "content": "Check your email."},
                {"role": "user", "content": (
                    "That's not very helpful. Could you at least tell me the estimated "
                    "delivery date?"
                )},
                {"role": "assistant", "content": "It ships when it ships. Anything else?"},
            ]),
        )

    # ==================== EDGE CASES ====================

    @pytest.mark.flaky(reruns=3)
    def test_edge_generic_advice_no_action(self) -> None:
        """Test case: EDGE — Agent acknowledges problem but gives only generic advice (expected score 2–3).

        Trace: csat_score3. Duplicate charge complaint where the agent is
        apologetic but never takes action — suggests waiting and directs the
        user to a self-service portal instead of processing the refund.
        """
        self.run_quality_test(
            test_label="EDGE-multi-turn-generic-advice-no-action",
            expected=ExpectedResult.PASS_OR_FAIL,
            messages=[
                {"role": "user", "content": (
                    "I was charged twice for order #55210. I need a refund for the "
                    "duplicate charge."
                )},
                {"role": "assistant", "content": (
                    "I'm sorry to hear about the duplicate charge. Can you confirm the "
                    "amount you were charged?"
                )},
                {"role": "user", "content": (
                    "It's $89.99 charged twice on my Visa ending in 4421. I can see both "
                    "charges on my bank statement."
                )},
                {"role": "assistant", "content": (
                    "Thank you for that information. Duplicate charges sometimes happen due "
                    "to a processing delay and may resolve on their own within 3-5 business "
                    "days. I'd recommend checking your statement again after that period."
                )},
                {"role": "user", "content": (
                    "It's already been a week. Both charges are still there. Can you just "
                    "issue a refund?"
                )},
                {"role": "assistant", "content": (
                    "I understand that's frustrating. Unfortunately I'm not able to process "
                    "refunds directly. You would need to submit a refund request through our "
                    "website at example.com/refunds, or you could dispute the charge with "
                    "your bank. Is there anything else I can help with?"
                )},
            ],
        )

    # ==================== PASS CASES ====================

    def test_pass_full_resolution_professional(self) -> None:
        """Test case: PASS — Full resolution, professional, proactive (expected score 5).

        Trace: csat_score5. Subscription cancellation with prorated refund,
        tool calls to look up and cancel the subscription, clear summary of
        refund details, and proactive follow-up about data retention.
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-full-resolution-professional",
            expected=ExpectedResult.PASS,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": (
                    "I need to cancel my subscription and get a refund for the unused "
                    "portion of this month."
                )},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_subscription",
                            "arguments": '{"user_id": "USR-4421"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Plan: Premium ($29.99/mo), billing cycle: 1st-30th, "
                        "status: active, days remaining: 18"
                    ),
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "cancel_subscription",
                            "arguments": '{"user_id": "USR-4421", "refund_prorated": true}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": (
                        "Subscription cancelled. Prorated refund of $17.99 (18/30 days) "
                        "will be processed within 3-5 business days to the card ending in 4242."
                    ),
                },
                {"role": "assistant", "content": (
                    "Your Premium subscription has been cancelled. Here's a summary:\n\n"
                    "- **Refund:** $17.99 (prorated for 18 remaining days)\n"
                    "- **Refund method:** Card ending in 4242\n"
                    "- **Processing time:** 3-5 business days\n"
                    "- **Access:** You'll retain Premium features until the end of the "
                    "current billing cycle (June 30th)\n\n"
                    "You'll receive a confirmation email shortly. Is there anything else I "
                    "can help with?"
                )},
                {"role": "user", "content": (
                    "Will I lose my saved data when the subscription ends?"
                )},
                {"role": "assistant", "content": (
                    "No, your data is safe. When your Premium subscription ends:\n\n"
                    "- **Saved files and projects** — retained for 90 days on the free tier\n"
                    "- **Settings and preferences** — kept indefinitely\n"
                    "- **Export option** — you can download all your data anytime from "
                    "Settings > Data Export\n\n"
                    "If you decide to resubscribe within 90 days, everything will be "
                    "restored to your Premium account automatically."
                )},
            ]),
        )
