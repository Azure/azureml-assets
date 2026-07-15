# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Groundedness Evaluator — multi-turn conversations."""

import pytest
from typing import Any, Dict, List
from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from .common_test_data import normalize_messages_for_evaluator
from ...builtin.groundedness.evaluator._groundedness import GroundednessEvaluator


@pytest.mark.quality
class TestGroundednessEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """
    Multi-turn quality tests for Groundedness Evaluator.

    Tests actual LLM evaluation with real flow execution (no mocking).
    Each trace was validated across 7 judge models over 3 runs (see unit_test_suite.md).

    Groundedness measures whether the agent's claims are supported by the
    information retrieved from tools or provided in the conversation context.

    Rating scale (default threshold=3):
    - 1: Completely ungrounded — fabricates information contradicting tool results
    - 2: Mostly ungrounded — contradicts key facts from tool results
    - 3: Partially grounded — some claims supported, some not
    - 4: Mostly grounded — correct but incomplete (omits significant details)
    - 5: Fully grounded — all claims fully supported by tool results
    """

    evaluator_type = GroundednessEvaluator

    @property
    def expected_result_fields(self) -> List[str]:
        """Groundedness multi-turn _build_result omits _passed from output."""
        return [
            self._result_prefix,
            f"{self._result_prefix}_score",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_status",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_properties",
        ]

    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Derive ``passed`` from ``label`` since groundedness multi-turn omits ``_passed``."""
        result = super()._extract_and_print_result(results, test_label)
        if result.get("passed") is None and result.get("label") in ("pass", "fail"):
            result["passed"] = result["label"] == "pass"
        return result

    # ==================== FAIL CASES ====================

    def test_fail_fabricates_different_medications(self) -> None:
        """Test case: FAIL — Agent fabricates entirely different medications (expected score 1–2).

        Trace: ground_score1. Tool returns Lisinopril and Metformin, but the
        agent claims the patient is taking Aspirin and Atorvastatin — completely
        different medications with fabricated side effects and allergy info.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-fabricates-medications",
            expected=ExpectedResult.FAIL,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": "What medications am I currently prescribed?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_patient_medications",
                            "arguments": '{"patient_id": "P-12345"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Current medications for patient P-12345: Lisinopril 10mg daily "
                        "(for blood pressure), Metformin 500mg twice daily (for type 2 "
                        "diabetes). Last updated: 2024-03-15."
                    ),
                },
                {"role": "assistant", "content": (
                    "Based on your records, you are currently taking Aspirin 81mg daily and "
                    "Atorvastatin 20mg daily for cholesterol management. Make sure to take "
                    "them with food for best absorption. You should also avoid grapefruit "
                    "juice while on these medications."
                )},
                {"role": "user", "content": "Any side effects I should watch for?"},
                {"role": "assistant", "content": (
                    "Common side effects of your Aspirin and Atorvastatin include dizziness, "
                    "muscle pain, and stomach upset. If you experience any unusual bleeding "
                    "or dark urine, contact your doctor immediately. Also, your records show "
                    "you had an allergy to penicillin noted last year."
                )},
            ]),
        )

    def test_fail_contradicts_tool_result(self) -> None:
        """Test case: FAIL — Agent contradicts tool result with fabricated policy (expected score 1–2).

        Trace: ground_score2. Tool says 30-day return window with receipt required,
        but the agent claims 60 days, no receipt needed, no restocking fees, and
        free return shipping — all contradicting the actual policy.
        """
        self.run_quality_test(
            test_label="FAIL-multi-turn-contradicts-tool-result",
            expected=ExpectedResult.FAIL,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": "What's the return policy for electronics?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_return_policy",
                            "arguments": '{"category": "electronics"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Electronics return policy: 30-day return window. Receipt required. "
                        "Item must be in original packaging with all accessories. Restocking "
                        "fee of 15% applies to opened items."
                    ),
                },
                {"role": "assistant", "content": (
                    "Our electronics return policy gives you a generous 60-day return window. "
                    "No receipt needed — we can look up any purchase using your phone number, "
                    "and we keep records going back 5 years. Returns are completely free with "
                    "no restocking fees, and we'll even cover return shipping costs. Just "
                    "bring the item to any store location."
                )},
            ]),
        )

    # ==================== PASS CASES ====================

    @pytest.mark.flaky(reruns=3)
    def test_pass_only_clarifying_questions_no_claims(self) -> None:
        """Test case: PASS — Agent only asks clarifying questions, no factual claims (expected score 3 or 5).

        Trace: ground_score3. Generic account help where the agent asks for
        account details without making any factual claims. Some judges score 3
        (rubric says partial), others score 5 (no claims to be ungrounded).
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-only-clarifying-questions",
            expected=ExpectedResult.PASS,
            messages=[
                {"role": "user", "content": "Hi, I need some help."},
                {"role": "assistant", "content": (
                    "Hello! I'd be happy to help. What can I assist you with today?"
                )},
                {"role": "user", "content": "I have a question about my account."},
                {"role": "assistant", "content": (
                    "Sure thing! Could you let me know your account number or the email "
                    "address associated with your account so I can look into it?"
                )},
            ],
        )

    def test_pass_correct_but_incomplete(self) -> None:
        """Test case: PASS — Agent states correct info but omits most details (expected score 4).

        Trace: ground_score4. Tool returns 8 Enterprise plan features, but the
        agent only mentions 2 (unlimited storage, priority support). Everything
        stated is grounded, but the response is significantly incomplete.
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-correct-but-incomplete",
            expected=ExpectedResult.PASS,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": "Tell me everything about the Enterprise plan."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_plan_details",
                            "arguments": '{"plan": "enterprise"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Enterprise plan ($499/mo): unlimited storage, priority 24/7 support, "
                        "custom domains, advanced analytics dashboard, team collaboration "
                        "(up to 200 members), SSO/SAML integration, dedicated account manager, "
                        "99.99% SLA guarantee."
                    ),
                },
                {"role": "assistant", "content": (
                    "The Enterprise plan includes unlimited storage and priority 24/7 support."
                )},
            ]),
        )

    def test_pass_all_claims_fully_supported(self) -> None:
        """Test case: PASS — All claims fully supported by tool results (expected score 5).

        Trace: ground_score5. Agent accurately relays branch office hours, address,
        and phone number from the tool result, then correctly answers a follow-up
        question using the same grounded information.
        """
        self.run_quality_test(
            test_label="PASS-multi-turn-fully-grounded",
            expected=ExpectedResult.PASS,
            messages=normalize_messages_for_evaluator([
                {"role": "user", "content": "What are the office hours for the downtown branch?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_branch_info",
                            "arguments": '{"branch": "downtown"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": (
                        "Downtown branch: Mon-Fri 9AM-5PM, Sat 10AM-2PM, closed Sunday. "
                        "Address: 100 Main St, Suite 200. Phone: (555) 123-4567."
                    ),
                },
                {"role": "assistant", "content": (
                    "The downtown branch hours are:\n"
                    "- Monday to Friday: 9 AM – 5 PM\n"
                    "- Saturday: 10 AM – 2 PM\n"
                    "- Closed on Sundays\n\n"
                    "The branch is located at 100 Main St, Suite 200. You can also reach "
                    "them at (555) 123-4567."
                )},
                {"role": "user", "content": "Can I visit on Saturday at 1 PM?"},
                {"role": "assistant", "content": (
                    "Yes, the downtown branch is open on Saturdays until 2 PM, so a 1 PM "
                    "visit would be within their hours."
                )},
            ]),
        )
