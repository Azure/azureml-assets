# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import pytest
from unittest.mock import MagicMock

from ...builtin.retrieval.evaluator._retrieval import RetrievalEvaluator
from ..common.evaluator_mock_config import create_none_score_flow_side_effect, create_mocked_evaluator, assert_none_score_result


# region None score handling tests

@pytest.mark.unittest
class TestRetrievalNoneScoreHandling:
    """Tests for None score handling in _do_eval (math.isnan fix).

    When _return_not_applicable_result returns score=None, _do_eval must not
    crash on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow should not raise TypeError."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = MagicMock(side_effect=create_none_score_flow_side_effect())
        result = evaluator(
            query="What are the office hours?",
            context="The office is open Monday through Friday from 9 AM to 5 PM.",
        )
        assert_none_score_result(result, "retrieval")


# endregion
