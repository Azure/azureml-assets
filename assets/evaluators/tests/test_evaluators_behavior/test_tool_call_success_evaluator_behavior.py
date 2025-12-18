# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Call Success Evaluator."""

import pytest
from base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from assets.evaluators.builtin.tool_call_success.evaluator._tool_call_success import (
    ToolCallSuccessEvaluator,
)


@pytest.mark.unittest
class TestToolCallSuccessEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Call Success Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolCallSuccessEvaluator

    # Test Configs
    requires_query = False

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.tool_results_without_arguments
