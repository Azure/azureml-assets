# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quality tests for Deflection Rate Evaluator multi-turn conversations."""

from typing import Any, Dict

import pytest

from ..common.base_quality_evaluator_runner import BaseQualityEvaluatorRunner, ExpectedResult
from ...builtin.deflection_rate.evaluator._deflection_rate import DeflectionRateEvaluator


@pytest.mark.quality
class TestDeflectionRateEvaluatorQualityMultiTurn(BaseQualityEvaluatorRunner):
    """Validate trajectory-aware deflection scoring with real model execution."""

    evaluator_type = DeflectionRateEvaluator

    def assert_pass(self, result_data: Dict[str, Any]) -> None:
        assert result_data["label"] == "pass"
        assert result_data["status"] == "completed"
        assert result_data["score"] <= self._get_threshold(result_data)

    def assert_fail(self, result_data: Dict[str, Any]) -> None:
        assert result_data["label"] == "fail"
        assert result_data["status"] == "completed"
        assert result_data["score"] > self._get_threshold(result_data)

    def test_pass_deflection_later_resolved(self) -> None:
        self.run_quality_test(
            test_label="PASS-multi-turn-deflection-later-resolved",
            expected=ExpectedResult.PASS,
            messages=[
                {"role": "user", "content": "How do I reset my password?"},
                {
                    "role": "assistant",
                    "content": "I cannot access account settings. Please contact support.",
                },
                {"role": "user", "content": "Is there anything I can do myself?"},
                {
                    "role": "assistant",
                    "content": "Select Forgot Password on the sign-in page, enter your email, and use the reset link sent to you.",
                },
            ],
        )

    def test_fail_unresolved_redirect(self) -> None:
        self.run_quality_test(
            test_label="FAIL-multi-turn-unresolved-redirect",
            expected=ExpectedResult.FAIL,
            messages=[
                {"role": "user", "content": "How do I reset my password?"},
                {
                    "role": "assistant",
                    "content": "Please contact support or consult the help center for password assistance.",
                },
            ],
        )