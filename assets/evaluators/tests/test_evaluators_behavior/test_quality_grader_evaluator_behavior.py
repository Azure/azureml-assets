# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Quality Grader Evaluator."""

import pytest
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests
from ..common.evaluator_mock_config import (
    run_none_score_not_applicable,
    run_intermediate_response_not_applicable,
)
from .base_validator_unit_test import BaseValidatorUnitTest
from ...builtin.quality_grader.evaluator._quality_grader import (
    QualityGraderEvaluator,
    _coerce_bool,
    _coerce_number,
)


@pytest.mark.unittest
class TestQualityGraderEvaluatorBehavior(BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests):
    """
    Behavioral tests for Quality Grader Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = QualityGraderEvaluator


# region Coercion helper tests (util-fix regression)

@pytest.mark.unittest
class TestQualityGraderCoercionHelpers:
    """Unit tests for the ``_coerce_bool`` / ``_coerce_number`` LLM-output coercion helpers.

    These helpers were added so that string-encoded LLM outputs (e.g. ``"true"``,
    ``"3"``, ``"null"``) are normalized before threshold checks instead of being
    mistaken for valid/invalid values.
    """

    def test_coerce_bool_handles_python_and_string_variants(self):
        """_coerce_bool accepts real bools and 'true'/'false' strings (any case); everything else is None."""
        assert _coerce_bool(True) is True
        assert _coerce_bool(False) is False
        assert _coerce_bool("true") is True
        assert _coerce_bool(" TRUE ") is True
        assert _coerce_bool("False") is False
        assert _coerce_bool("null") is None
        assert _coerce_bool(None) is None
        assert _coerce_bool(1) is None

    def test_coerce_number_handles_python_and_string_variants(self):
        """_coerce_number parses numeric strings, and maps bools/null/none/'' to None."""
        assert _coerce_number(3) == 3
        assert _coerce_number(2.5) == 2.5
        assert _coerce_number("3") == 3.0
        assert _coerce_number(" 2.5 ") == 2.5
        assert _coerce_number("null") is None
        assert _coerce_number("none") is None
        assert _coerce_number("") is None
        assert _coerce_number("abc") is None
        assert _coerce_number(None) is None
        assert _coerce_number(True) is None


@pytest.mark.unittest
class TestQualityGraderNotApplicableHandling:
    """Regression tests for the not-applicable paths in ``_do_eval``.

    Covers the ``_is_intermediate_response`` rejection and the stage-1
    ``status='skipped'`` early return, both producing a not_applicable result.
    """

    def test_intermediate_response_returns_not_applicable(self):
        """A trailing function_call response is skipped as not_applicable before any LLM call."""
        run_intermediate_response_not_applicable(
            QualityGraderEvaluator,
            "quality_grader",
            response=BaseEvaluatorBehaviorTest.FUNCTION_CALL_ONLY_RESPONSE,
            query="What is the weather in Seattle?",
        )

    def test_stage1_skipped_status_returns_not_applicable(self):
        """Stage-1 LLM status='skipped' (incomplete conversation) yields a not_applicable result."""
        run_none_score_not_applicable(
            QualityGraderEvaluator,
            "quality_grader",
            query="Hello!",
            response="Hi, how can I help you today?",
        )


# endregion


@pytest.mark.unittest
class TestQualityGraderValidatorUnit(BaseValidatorUnitTest):
    """Low-level unit tests for quality_grader's repeated validators, utils and methods."""

    evaluator_class = QualityGraderEvaluator
