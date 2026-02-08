# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Input Accuracy Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from ...builtin.tool_input_accuracy.evaluator._tool_input_accuracy import (
    ToolInputAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolInputAccuracyEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Input Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolInputAccuracyEvaluator

    # Test Configs
    requires_tool_definitions = True

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_calls_with_arguments

    _additional_expected_field_suffixes = ["details"]
