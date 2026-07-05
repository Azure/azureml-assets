# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Quality Grader Evaluator."""

import pytest
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests
from ..common.evaluator_mock_config import (
    run_none_score_not_applicable,
    run_intermediate_response_not_applicable,
)
from .base_validator_unit_test import (
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    LogSafeSummaryUnitTests,
    MessagePreprocessUnitTests,
)
from ...builtin.quality_grader.evaluator._quality_grader import (
    QualityGraderEvaluator,
    _coerce_bool,
    _coerce_number,
)


def _build_quality_grader_with_flows(stage1_output, stage2_output=None):
    """Build a QualityGraderEvaluator with its two prompty flows mocked.

    :param stage1_output: The dict returned by the stage-1 (response quality) flow.
    :param stage2_output: The dict returned by the stage-2 (groundedness) flow.
    :return: The evaluator with ``_flow`` and ``_groundedness_flow`` mocked.
    """
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint="https://Sanitized.api.cognitive.microsoft.com",
        azure_deployment="aoai-deployment",
    )
    evaluator = QualityGraderEvaluator(model_config=model_config)

    async def _stage1_flow(timeout=None, **kwargs):
        return stage1_output

    async def _stage2_flow(timeout=None, **kwargs):
        return stage2_output

    evaluator._flow = MagicMock(side_effect=_stage1_flow)
    evaluator._groundedness_flow = MagicMock(side_effect=_stage2_flow)
    if hasattr(evaluator, "_ensure_query_prompty_loaded"):
        evaluator._ensure_query_prompty_loaded = MagicMock()
    return evaluator


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
class TestQualityGraderTwoStagePipeline:
    """Regression tests for the two-stage grading pipeline in ``_do_eval`` and ``_build_result``.

    Drives the stage-1 threshold failures, the stage-2 groundedness checks
    (numeric, null, and pass paths), and the JSON output parsing branches.
    """

    STAGE1_PASS = {
        "properties": {"abstention": False, "relevance": 5, "answerCompleteness": 5},
        "status": "completed",
        "reasoning": "The response is relevant and complete.",
        "score": 1,
    }

    def test_stage1_threshold_failure_returns_fail_result(self):
        """Stage-1 abstention/low relevance/low completeness produce a failing result."""
        evaluator = _build_quality_grader_with_flows(
            {
                "properties": {"abstention": True, "relevance": 1, "answerCompleteness": 1},
                "status": "completed",
                "reasoning": "",
            }
        )
        result = evaluator(query="What is the capital of France?", response="I cannot help.")
        assert isinstance(result, dict)
        assert result["quality_grader"] == 0.0
        assert evaluator._groundedness_flow.call_count == 0

    def test_stage2_numeric_failure_returns_fail_result(self):
        """Stage-2 numeric groundedness/context-coverage below threshold fail the result."""
        evaluator = _build_quality_grader_with_flows(
            self.STAGE1_PASS,
            {
                "properties": {"groundedness": 1, "contextCoverage": 1},
                "status": "completed",
                "reasoning": "Not grounded.",
            },
        )
        result = evaluator(query="Q", response="R", context="Some retrieved context.")
        assert result["quality_grader"] == 0.0
        assert evaluator._groundedness_flow.call_count == 1

    def test_stage2_null_values_return_fail_result(self):
        """Stage-2 null groundedness/context-coverage exercise the None fallback branches."""
        evaluator = _build_quality_grader_with_flows(
            self.STAGE1_PASS,
            {
                "properties": {"groundedness": None, "contextCoverage": None},
                "status": "completed",
                "reasoning": "",
            },
        )
        result = evaluator(query="Q", response="R", context="Some retrieved context.")
        assert result["quality_grader"] == 0.0

    def test_stage2_pass_returns_pass_result(self):
        """A passing stage-1 and stage-2 produce a passing result with stage-2 properties."""
        evaluator = _build_quality_grader_with_flows(
            self.STAGE1_PASS,
            {
                "properties": {"groundedness": 5, "contextCoverage": 5, "documentUtility": 5},
                "status": "completed",
                "reasoning": "Fully grounded.",
            },
        )
        result = evaluator(query="Q", response="R", context="Some retrieved context.")
        assert result["quality_grader"] == 1.0

    def test_parse_prompty_json_output_branches(self):
        """Parse JSON output across empty, dict, and malformed-string branches."""
        parse = QualityGraderEvaluator._parse_prompty_json_output
        assert parse(None) == {}
        assert parse({"llm_output": ""}) == {}
        assert parse({"llm_output": {"score": 1}}) == {"score": 1}
        assert parse({"llm_output": "not-json"}) == {}


@pytest.mark.unittest
class TestQualityGraderValidatorUnit(
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    LogSafeSummaryUnitTests,
):
    """Low-level unit tests for quality_grader's repeated validators, utils and methods."""

    evaluator_class = QualityGraderEvaluator
