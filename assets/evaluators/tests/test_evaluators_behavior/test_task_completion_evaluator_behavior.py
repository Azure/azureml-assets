# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Completion Evaluator."""

import pytest
from base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from assets.evaluators.builtin.task_completion.evaluator._task_completion import (
    TaskCompletionEvaluator,
)


@pytest.mark.unittest
class TestTaskCompletionEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Task Completion Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = TaskCompletionEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response
