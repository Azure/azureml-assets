# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Customer Satisfaction Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.customer_satisfaction.evaluator._customer_satisfaction import (
    CustomerSatisfactionEvaluator,
)


@pytest.mark.unittest
class TestCustomerSatisfactionEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Customer Satisfaction Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = CustomerSatisfactionEvaluator

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.MINIMAL_RESPONSE

    _additional_expected_field_suffixes = ["dimensions"]

    @property
    def expected_result_fields(self):
        """Get the expected result fields for customer satisfaction evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_dimensions",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]
