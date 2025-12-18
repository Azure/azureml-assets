# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Coherence Evaluator."""

import pytest
from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from assets.evaluators.builtin.coherence.evaluator._coherence import CoherenceEvaluator


@pytest.mark.unittest
class TestCoherenceEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Coherence Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = CoherenceEvaluator
