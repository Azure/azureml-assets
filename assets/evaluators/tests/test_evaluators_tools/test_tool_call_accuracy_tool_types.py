# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tool type tests for Tool Call Accuracy Evaluator.

Tests that the evaluator correctly processes conversations containing
various tool types (function, code_interpreter, bing_grounding, etc.)
from real converter output data.
"""

import pytest
from ..common.base_tool_type_evaluator_test import BaseToolTypeEvaluatorTest
from ...builtin.tool_call_accuracy.evaluator._tool_call_accuracy import (
    ToolCallAccuracyEvaluator,
)


@pytest.mark.unittest
class TestToolCallAccuracyToolTypes(BaseToolTypeEvaluatorTest):
    """
    Tool type tests for ToolCallAccuracyEvaluator.

    All test methods are inherited from BaseToolTypeEvaluatorTest.
    This class only sets the evaluator type.
    """

    evaluator_type = ToolCallAccuracyEvaluator
