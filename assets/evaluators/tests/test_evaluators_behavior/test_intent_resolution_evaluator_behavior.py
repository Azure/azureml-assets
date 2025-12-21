# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Intent Resolution Evaluator."""

import pytest
from base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from ...builtin.intent_resolution.evaluator._intent_resolution import (
    IntentResolutionEvaluator,
)


@pytest.mark.unittest
class TestIntentResolutionEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Intent Resolution Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = IntentResolutionEvaluator
