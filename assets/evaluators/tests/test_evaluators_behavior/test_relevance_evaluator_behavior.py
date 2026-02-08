# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Relevance Evaluator."""

import pytest
from typing import List
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator


@pytest.mark.unittest
class TestRelevanceEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Relevance Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = RelevanceEvaluator

    @property
    def expected_result_fields(self) -> List[str]:
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]
