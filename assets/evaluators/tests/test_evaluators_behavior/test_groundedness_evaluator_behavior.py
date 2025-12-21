# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Groundedness Evaluator."""

import pytest
from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.groundedness.evaluator._groundedness import (
    GroundednessEvaluator,
)


@pytest.mark.unittest
class TestGroundednessEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Groundedness Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = GroundednessEvaluator

    # Test Configs
    requires_query = False

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.weather_tool_call_and_assistant_response
