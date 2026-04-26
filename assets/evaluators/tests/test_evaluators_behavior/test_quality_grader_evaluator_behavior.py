# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Quality Grader Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.quality_grader.evaluator._quality_grader import QualityGraderEvaluator


@pytest.mark.unittest
class TestQualityGraderEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Quality Grader Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = QualityGraderEvaluator
