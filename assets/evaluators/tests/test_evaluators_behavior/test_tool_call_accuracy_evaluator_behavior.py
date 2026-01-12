# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Call Accuracy Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import pytest
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolCallAccuracyEvaluatorBehavior(BaseToolCallEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Call Accuracy Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolCallAccuracyEvaluator
    MINIMAL_RESPONSE = BaseToolCallEvaluatorBehaviorTest.email_tool_call_and_assistant_response
