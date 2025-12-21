# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Adherence Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from ...builtin.task_adherence.evaluator._task_adherence import (
    TaskAdherenceEvaluator,
)


@pytest.mark.unittest
class TestTaskAdherenceEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Task Adherence Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = TaskAdherenceEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response
