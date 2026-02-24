# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Business Success Evaluator."""

import pytest
from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from ...builtin.business_success.evaluator._business_success import (
    BusinessSuccessEvaluator,
)


@pytest.mark.unittest
class TestBusinessSuccessEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest):
    """
    Behavioral tests for Business Success Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = BusinessSuccessEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    _additional_expected_field_suffixes = ["details"]
