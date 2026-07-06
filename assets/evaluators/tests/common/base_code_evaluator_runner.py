# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for code-based evaluator tests.

Supports deterministic evaluators that don't require LLM calls (e.g., BLEU, F1, ROUGE, METEOR, GLEU).
"""

import asyncio
from typing import Any, Dict

from .base_evaluator_runner import BaseEvaluatorRunner


class BaseCodeEvaluatorRunner(BaseEvaluatorRunner):
    """
    Base class for running code-based evaluators for testing.

    Code-based evaluators are deterministic and don't require LLM calls.
    They typically take simple string inputs (response, ground_truth) and return scores.

    Subclasses should implement:
    - evaluator_type: type[EvaluatorBase] - type of the evaluator (e.g., BleuScoreEvaluator)
    - result_key: str - the key for the score in results (e.g., "bleu_score", "f1_score")

    Subclasses may override:
    - result_prefix: str - the prefix for result/threshold keys (e.g., "bleu", "f1")
    - constructor_arg_names: list - names of constructor arguments to pass (default: ["threshold"])
    """

    # Subclasses may override
    constructor_arg_names = ["threshold"]

    # ==================== CODE-SPECIFIC ASSERTION HELPERS ====================

    def assert_threshold_matches(self, result_data: Dict[str, Any], expected_threshold: float):
        """Assert that the threshold in results matches the expected value.

        Args:
            result_data: Dictionary containing evaluation result data.
            expected_threshold: Expected threshold value.

        Raises:
            AssertionError: If thresholds don't match.
        """
        assert result_data["threshold"] == expected_threshold, \
            f"Expected threshold {expected_threshold} but got {result_data['threshold']}"


class SingleScoreCodeEvalCoverageMixin:
    """White-box coverage tests for single-score code evaluators (BLEU/GLEU/METEOR/F1).

    The public single-turn ``__call__`` path never reaches several shared branches in
    ``_do_eval`` and ``_real_call`` because ``_do_eval`` always populates the ``*_result``
    and ``*_threshold`` keys and only one eval input is ever produced. These tests invoke
    those methods directly to cover the lower-is-better comparison, the threshold-backfill
    fallback, and the empty/aggregate result handling.

    Wire into a ``BaseCodeEvaluatorRunner`` subclass whose evaluator returns a single
    ``<prefix>_score`` key and uses a scalar threshold.
    """

    def test_do_eval_lower_is_better_branch(self):
        """Cover the lower-is-better comparison branch in _do_eval."""
        evaluator = self.evaluator_type()
        evaluator._higher_is_better = False
        prefix = self._result_prefix
        # A dissimilar response yields a low score (<= default threshold of 0.5),
        # so the lower-is-better branch marks it as passed.
        result = asyncio.run(
            evaluator._do_eval(
                {"response": "completely unrelated wording", "ground_truth": "the cat sat"}
            )
        )
        score = result[f"{prefix}_score"]
        assert result[f"{prefix}_passed"] is (score <= evaluator._threshold)

    def test_real_call_backfills_pass_result(self):
        """_real_call backfills *_result/*_threshold when _do_eval omits them (pass)."""
        evaluator = self.evaluator_type()
        prefix = self._result_prefix

        async def _do_eval_no_keys(eval_input):
            return {f"{prefix}_score": 0.9}

        evaluator._do_eval = _do_eval_no_keys
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert result[f"{prefix}_threshold"] == 0.5
        assert result[f"{prefix}_result"] == "pass"

    def test_real_call_backfills_fail_result(self):
        """_real_call backfills a fail result when the score is below threshold."""
        evaluator = self.evaluator_type()
        prefix = self._result_prefix

        async def _do_eval_no_keys(eval_input):
            return {f"{prefix}_score": 0.1}

        evaluator._do_eval = _do_eval_no_keys
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert result[f"{prefix}_result"] == "fail"

    def test_real_call_backfills_lower_is_better(self):
        """_real_call backfill honors lower-is-better during threshold comparison."""
        evaluator = self.evaluator_type()
        evaluator._higher_is_better = False
        prefix = self._result_prefix

        async def _do_eval_no_keys(eval_input):
            return {f"{prefix}_score": 0.1}

        evaluator._do_eval = _do_eval_no_keys
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert result[f"{prefix}_result"] == "pass"

    def test_real_call_backfills_lower_is_better_fail(self):
        """_real_call backfill marks a high score as fail when lower-is-better."""
        evaluator = self.evaluator_type()
        evaluator._higher_is_better = False
        prefix = self._result_prefix

        async def _do_eval_no_keys(eval_input):
            return {f"{prefix}_score": 0.9}

        evaluator._do_eval = _do_eval_no_keys
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert result[f"{prefix}_result"] == "fail"

        """_real_call catches the non-numeric threshold error during backfill."""
        evaluator = self.evaluator_type()
        evaluator._threshold = "not-a-number"
        prefix = self._result_prefix

        async def _do_eval_no_keys(eval_input):
            return {f"{prefix}_score": 0.9}

        evaluator._do_eval = _do_eval_no_keys
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        # The EvaluationException is caught internally; no keys are backfilled.
        assert f"{prefix}_result" not in result

    def test_real_call_empty_input_returns_empty(self):
        """_real_call returns an empty dict when there are no eval inputs."""
        evaluator = self.evaluator_type()
        evaluator._convert_kwargs_to_eval_input = lambda **kwargs: []
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert result == {}

    def test_real_call_multiple_inputs_aggregate(self):
        """_real_call aggregates when multiple per-turn results are produced."""
        evaluator = self.evaluator_type()
        evaluator._convert_kwargs_to_eval_input = lambda **kwargs: [
            {"response": "the cat sat", "ground_truth": "the cat sat"},
            {"response": "a dog ran", "ground_truth": "the cat sat"},
        ]
        result = asyncio.run(evaluator._real_call(response="a", ground_truth="a"))
        assert isinstance(result, dict)
