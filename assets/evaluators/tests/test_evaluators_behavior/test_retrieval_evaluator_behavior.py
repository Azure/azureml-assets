# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import pytest

from .base_validator_unit_test import BaseValidatorUnitTest
from ...builtin.retrieval.evaluator._retrieval import RetrievalEvaluator
from ..common.evaluator_mock_config import run_none_score_not_applicable


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
class TestRetrievalValidatorUnit(BaseValidatorUnitTest):
    """Low-level unit tests for retrieval's repeated validators, utils and methods."""

    evaluator_class = RetrievalEvaluator
