# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Behavioral tests for Tool Selection Evaluator using AIProjectClient.

Tests various input scenarios: query, response, tool_definitions, and tool_calls.
"""

import pytest
from base_tool_parameters_behavior_test import BaseToolParametersBehaviorTest
from assets.evaluators.builtin.tool_selection.evaluator._tool_selection import (
    ToolSelectionEvaluator,
)


@pytest.mark.unittest
class TestToolSelectionEvaluatorBehavior(BaseToolParametersBehaviorTest):
    """
    Behavioral tests for Tool Selection Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolSelectionEvaluator

    # Test Configs
    requires_valid_format = True
    requires_tool_definitions = True
    needs_arguments = False
