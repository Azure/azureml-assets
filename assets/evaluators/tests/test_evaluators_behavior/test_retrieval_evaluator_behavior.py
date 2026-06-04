# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import os
import pytest
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration

from ...builtin.retrieval.evaluator._retrieval import RetrievalEvaluator
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator, create_none_score_flow_side_effect


def _create_mocked_retrieval_evaluator():
    """Create a RetrievalEvaluator with _flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = RetrievalEvaluator(model_config=model_config)
    evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator("retrieval"))
    return evaluator


# region None score handling tests

@pytest.mark.unittest
class TestRetrievalNoneScoreHandling:
    """Tests for None score handling in _do_eval (math.isnan fix).

    When _return_not_applicable_result returns score=None, _do_eval must not
    crash on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow should not raise TypeError."""
        evaluator = _create_mocked_retrieval_evaluator()
        evaluator._flow = MagicMock(side_effect=create_none_score_flow_side_effect())
        result = evaluator(
            query="What are the office hours?",
            context="The office is open Monday through Friday from 9 AM to 5 PM.",
        )
        assert result["retrieval"] is None
        assert result["retrieval_result"] == "not_applicable"


# endregion
