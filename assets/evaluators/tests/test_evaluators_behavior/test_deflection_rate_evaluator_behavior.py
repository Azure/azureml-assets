# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Deflection Rate Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.deflection_rate.evaluator._deflection_rate import (
    DeflectionRateEvaluator,
)


@pytest.mark.unittest
class TestDeflectionRateEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Deflection Rate Evaluator.

    Tests different input formats and scenarios.
    Note: This evaluator only requires response, not query.
    """

    evaluator_type = DeflectionRateEvaluator

    # Deflection rate only needs response, not query
    requires_query = False

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.MINIMAL_RESPONSE

    _additional_expected_field_suffixes = ["deflection_type"]

    @property
    def expected_result_fields(self):
        """Get the expected result fields for deflection rate evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_deflection_type",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]
