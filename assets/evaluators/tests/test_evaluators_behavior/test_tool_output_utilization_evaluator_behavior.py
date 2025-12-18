# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Tool Output Utilization Evaluator."""

import pytest
from base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from assets.evaluators.builtin.tool_output_utilization.evaluator._tool_output_utilization import (
    ToolOutputUtilizationEvaluator,
)


@pytest.mark.unittest
class TestToolOutputUtilizationEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Tool Output Utilization Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = ToolOutputUtilizationEvaluator

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.VALID_RESPONSE
