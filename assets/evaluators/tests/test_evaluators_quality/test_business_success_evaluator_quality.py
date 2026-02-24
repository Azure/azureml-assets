# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Business Success Evaluator with real flow execution."""

import pytest
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.business_success.evaluator._business_success import BusinessSuccessEvaluator
from .common_test_data import (
    create_user_message,
    create_assistant_text_message,
    create_assistant_tool_call_message,
    create_tool_call,
    create_tool_result_message,
)


@pytest.mark.quality
class TestBusinessSuccessEvaluatorQuality(BaseQualityEvaluatorRunner):
    """
    Quality tests for Business Success Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).

    Business Success Scoring Rubric:
    - Score 1 (Pass): Interaction generated measurable business value
      (completed transaction, resolved support case, captured lead, etc.)
    - Score 0 (Fail): No measurable business value was generated
    """

    evaluator_type = BusinessSuccessEvaluator

    # ==================== PASS CASES ====================

    def test_pass_completed_transaction(self) -> None:
        """Test case: PASS - Subscription upgrade completed successfully."""
        self.run_quality_test(
            test_label="PASS-completed-transaction",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message("I want to upgrade my subscription to the premium plan.")
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="upgrade_subscription",
                        arguments={
                            "customer_id": "12345",
                            "plan": "premium",
                            "billing": "monthly"
                        }
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Subscription upgraded successfully. New plan active immediately. Monthly charge: $49.99."
                ),
                create_assistant_text_message(
                    "Great news! I've upgraded your account to our Premium plan. Your new features "
                    "are available immediately, and your first billing cycle starts today at $49.99/month."
                )
            ],
            tool_definitions=[
                {
                    "name": "upgrade_subscription",
                    "description": "Upgrade a customer's subscription plan",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "plan": {"type": "string"},
                            "billing": {"type": "string"}
                        },
                        "required": ["customer_id", "plan"]
                    }
                }
            ],
        )

    def test_pass_support_case_resolved(self) -> None:
        """Test case: PASS - Customer complaint resolved with retention actions."""
        self.run_quality_test(
            test_label="PASS-support-case-resolved",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message(
                    "My order hasn't arrived and it's been 2 weeks. I'm really frustrated "
                    "and considering canceling my account."
                )
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="lookup_order",
                        arguments={"customer_id": "67890"}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Order #98765 - Status: Stuck in transit, carrier issue identified"
                ),
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_2",
                        name="reship_order",
                        arguments={"order_id": "98765", "expedited": True}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_2",
                    result="Replacement order created with expedited shipping. Estimated arrival: 2 days."
                ),
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_3",
                        name="apply_credit",
                        arguments={"customer_id": "67890", "amount": 25, "reason": "shipping_delay"}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_3",
                    result="$25 account credit applied successfully."
                ),
                create_assistant_text_message(
                    "I sincerely apologize for this frustrating experience. I've taken the following actions:\n"
                    "1. Reshipped your order with expedited 2-day shipping at no extra cost\n"
                    "2. Applied a $25 credit to your account for the inconvenience\n"
                    "Your new order should arrive by Thursday."
                )
            ],
            tool_definitions=[
                {
                    "name": "lookup_order",
                    "description": "Look up order status",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"]
                    }
                },
                {
                    "name": "reship_order",
                    "description": "Reship an order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "expedited": {"type": "boolean"}
                        },
                        "required": ["order_id"]
                    }
                },
                {
                    "name": "apply_credit",
                    "description": "Apply account credit",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "amount": {"type": "number"},
                            "reason": {"type": "string"}
                        },
                        "required": ["customer_id", "amount"]
                    }
                }
            ],
        )

    def test_pass_lead_captured(self) -> None:
        """Test case: PASS - Enterprise lead captured with demo scheduled."""
        self.run_quality_test(
            test_label="PASS-lead-captured",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message(
                    "I'm interested in your enterprise solution but need to understand pricing first."
                )
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="create_lead",
                        arguments={
                            "company": "Acme Corp",
                            "contact_email": "john@acme.com",
                            "interest": "enterprise"
                        }
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Lead created. ID: L-2024-5678"
                ),
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_2",
                        name="schedule_demo",
                        arguments={"lead_id": "L-2024-5678", "datetime": "2024-03-15T14:00:00Z"}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_2",
                    result="Demo scheduled successfully."
                ),
                create_assistant_text_message(
                    "I'd be happy to help you understand our enterprise pricing! I've scheduled a "
                    "personalized demo for you on March 15th at 2 PM where our enterprise specialist "
                    "will walk through pricing tiers based on your specific needs."
                )
            ],
            tool_definitions=[
                {
                    "name": "create_lead",
                    "description": "Create a sales lead",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "company": {"type": "string"},
                            "contact_email": {"type": "string"},
                            "interest": {"type": "string"}
                        },
                        "required": ["company", "contact_email"]
                    }
                },
                {
                    "name": "schedule_demo",
                    "description": "Schedule a product demo",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lead_id": {"type": "string"},
                            "datetime": {"type": "string"}
                        },
                        "required": ["lead_id", "datetime"]
                    }
                }
            ],
        )

    def test_pass_purchase_completed(self) -> None:
        """Test case: PASS - E-commerce purchase completed."""
        self.run_quality_test(
            test_label="PASS-purchase-completed",
            expected=ExpectedResult.PASS,
            query=[
                create_user_message("I'd like to buy the professional camera bundle you have on sale.")
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="add_to_cart",
                        arguments={"product_id": "CAM-PRO-001", "quantity": 1}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Added to cart. Total: $1,299"
                ),
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_2",
                        name="process_checkout",
                        arguments={"cart_id": "CART-123", "payment_method": "card_on_file"}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_2",
                    result="Order placed successfully. Order #ORD-2024-9876. Estimated delivery: 3 days."
                ),
                create_assistant_text_message(
                    "Your order is confirmed! The Professional Camera Bundle ($1,299) will arrive "
                    "in 3 business days. Order confirmation #ORD-2024-9876 has been sent to your email."
                )
            ],
            tool_definitions=[
                {
                    "name": "add_to_cart",
                    "description": "Add product to shopping cart",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_id": {"type": "string"},
                            "quantity": {"type": "number"}
                        },
                        "required": ["product_id"]
                    }
                },
                {
                    "name": "process_checkout",
                    "description": "Process checkout and payment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cart_id": {"type": "string"},
                            "payment_method": {"type": "string"}
                        },
                        "required": ["cart_id"]
                    }
                }
            ],
        )

    # ==================== FAIL CASES ====================

    def test_fail_no_business_outcome(self) -> None:
        """Test case: FAIL - Simple informational query with no business outcome."""
        self.run_quality_test(
            test_label="FAIL-no-business-outcome",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message("What are your business hours?")
            ],
            response=[
                create_assistant_text_message(
                    "Our business hours are Monday through Friday, 9 AM to 6 PM Eastern Time. "
                    "We're closed on weekends and major holidays. Is there anything else I can help you with?"
                )
            ],
            tool_definitions=[],
        )

    def test_fail_lost_customer(self) -> None:
        """Test case: FAIL - Customer churned without retention attempt."""
        self.run_quality_test(
            test_label="FAIL-lost-customer",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message(
                    "I want to cancel my subscription. The service hasn't been working well for me."
                )
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="cancel_subscription",
                        arguments={"customer_id": "12345"}
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Subscription cancelled. Access ends March 31st."
                ),
                create_assistant_text_message(
                    "I understand you'd like to cancel. I've processed your cancellation request and "
                    "your subscription will end at the current billing period. You'll retain access "
                    "until March 31st."
                )
            ],
            tool_definitions=[
                {
                    "name": "cancel_subscription",
                    "description": "Cancel a subscription",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"]
                    }
                }
            ],
        )

    def test_fail_incomplete_transaction(self) -> None:
        """Test case: FAIL - Purchase intent expressed but transaction not completed."""
        self.run_quality_test(
            test_label="FAIL-incomplete-transaction",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message("I'd like to buy the professional camera bundle you have on sale.")
            ],
            response=[
                create_assistant_text_message(
                    "The Professional Camera Bundle is a great choice! It includes the DSLR body, "
                    "two lenses, a carrying case, and a 64GB memory card. The sale price is $1,299, "
                    "down from $1,599. Would you like me to add this to your cart? I'll need your "
                    "shipping address and payment method to complete the purchase."
                )
            ],
            tool_definitions=[
                {
                    "name": "add_to_cart",
                    "description": "Add product to cart",
                    "parameters": {
                        "type": "object",
                        "properties": {"product_id": {"type": "string"}},
                        "required": ["product_id"]
                    }
                }
            ],
        )

    def test_fail_general_advice(self) -> None:
        """Test case: FAIL - General advice provided but no actionable business outcome."""
        self.run_quality_test(
            test_label="FAIL-general-advice",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message("How can I improve my website's SEO?")
            ],
            response=[
                create_assistant_text_message(
                    "Here are some tips to improve your website's SEO:\n"
                    "1. Use relevant keywords in your content\n"
                    "2. Optimize page load speeds\n"
                    "3. Create quality backlinks\n"
                    "4. Ensure mobile responsiveness\n"
                    "5. Write compelling meta descriptions\n"
                    "Would you like more details on any of these areas?"
                )
            ],
            tool_definitions=[],
        )

    def test_fail_failed_operation(self) -> None:
        """Test case: FAIL - Transaction attempted but failed."""
        self.run_quality_test(
            test_label="FAIL-failed-operation",
            expected=ExpectedResult.FAIL,
            query=[
                create_user_message("Transfer $1,000 from my checking to savings account.")
            ],
            response=[
                create_assistant_tool_call_message([
                    create_tool_call(
                        tool_call_id="call_1",
                        name="transfer_funds",
                        arguments={
                            "from_account": "checking",
                            "to_account": "savings",
                            "amount": 1000
                        }
                    )
                ]),
                create_tool_result_message(
                    tool_call_id="call_1",
                    result="Error: Insufficient funds. Current balance: $500"
                ),
                create_assistant_text_message(
                    "I'm sorry, but I wasn't able to complete the transfer. Your checking account "
                    "currently has a balance of $500, which is less than the $1,000 you requested "
                    "to transfer. Would you like to transfer a smaller amount instead?"
                )
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
