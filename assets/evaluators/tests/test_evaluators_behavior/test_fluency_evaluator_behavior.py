# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""
Behavioral tests for Fluency Evaluator.
"""

import pytest
from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from assets.evaluators.builtin.fluency.evaluator._fluency import FluencyEvaluator


@pytest.mark.unittest
class TestFluencyEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Fluency Evaluator.
    Tests different input formats and scenarios.
    """

    evaluator_type = FluencyEvaluator

    # Test Configs
    requires_query = False
