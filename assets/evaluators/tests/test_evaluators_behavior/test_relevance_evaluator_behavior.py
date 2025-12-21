# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Relevance Evaluator."""

import pytest
from base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator


@pytest.mark.unittest
class TestRelevanceEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Relevance Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = RelevanceEvaluator
