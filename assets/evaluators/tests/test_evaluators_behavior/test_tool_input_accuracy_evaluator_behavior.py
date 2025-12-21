# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Input Accuracy Evaluator."""

import pytest
from base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    ToolInputAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolInputAccuracyEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Input Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolInputAccuracyEvaluator

    MINIMAL_RESPONSE = BaseToolCallEvaluatorBehaviorTest.tool_results_with_arguments
