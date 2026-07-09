# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import asyncio

import pytest

from .base_validator_unit_test import (
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ...builtin.retrieval.evaluator._retrieval import RetrievalEvaluator
from ..common.evaluator_mock_config import (
    INTERMEDIATE_FUNCTION_CALL_RESPONSE,
    create_mocked_evaluator,
    run_none_score_not_applicable,
)


# region None score handling tests

@pytest.mark.unittest
class TestRetrievalNoneScoreHandling:
    """Tests for None score handling in _do_eval (math.isnan fix).

    When _return_not_applicable_result returns score=None, _do_eval must not
    crash on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow should not raise TypeError."""
        run_none_score_not_applicable(
            RetrievalEvaluator,
            "retrieval",
            query="What are the office hours?",
            context="The office is open Monday through Friday from 9 AM to 5 PM.",
        )


# endregion


@pytest.mark.unittest
class TestRetrievalValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
):
    """Low-level unit tests for retrieval's repeated validators, utils and methods."""

    evaluator_class = RetrievalEvaluator


# region _do_eval override branch coverage

@pytest.mark.unittest
class TestRetrievalDoEvalBranches:
    """Cover retrieval's override ``_do_eval`` intermediate and list-preprocessing branches."""

    def test_intermediate_response_not_applicable(self):
        """An intermediate (function_call) response short-circuits to a not-applicable result."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        result = asyncio.run(evaluator._do_eval({"response": INTERMEDIATE_FUNCTION_CALL_RESPONSE}))
        assert result["retrieval_result"] == "not_applicable"

    def test_list_inputs_are_preprocessed(self):
        """List-typed query and response inputs are preprocessed before the flow call."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        result = asyncio.run(
            evaluator._do_eval(
                {
                    "query": [{"role": "user", "content": [{"type": "text", "text": "What are the hours?"}]}],
                    "response": [{"role": "assistant", "content": [{"type": "text", "text": "9 to 5."}]}],
                    "context": "The office is open 9 to 5.",
                }
            )
        )
        assert result["retrieval_score"] == 5
