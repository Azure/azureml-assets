# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Base class for code-based evaluator tests.

Supports deterministic evaluators that don't require LLM calls (e.g., BLEU, F1, ROUGE, METEOR, GLEU).
"""

import asyncio
import importlib
import json
from typing import Any, Dict

import pytest
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget

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

    @property
    def _evaluator_module(self):
        """Return the module that owns the evaluator implementation."""
        module = importlib.import_module(self.evaluator_type.__module__)
        if not hasattr(module, "_parse_response_for_evaluation"):
            pytest.skip("Evaluator does not support message response parsing")
        return module

    def test_message_parser_representations(self):
        """Cover string, JSON, direct-list, and structured-content parsing."""
        module = self._evaluator_module
        assert module._parse_response_for_evaluation("plain response") == "plain response"
        assert module._parse_response_for_evaluation("[{bad json") == "[{bad json"
        assert module._parse_response_for_evaluation('{"answer": "text"}') == '{"answer": "text"}'
        assert module._parse_response_for_evaluation(
            json.dumps([{"role": "assistant", "content": "final answer"}])
        ) == "final answer"
        assert module._parse_response_for_evaluation(
            [{"role": "assistant", "content": [{"type": "text", "text": "final answer"}]}]
        ) == "final answer"

    def test_message_extractor_skips_non_text_and_stops_at_user(self):
        """Cover ignored message types and the latest-user boundary."""
        module = self._evaluator_module
        assert module._extract_final_text_response([123, {"role": "narrator", "content": "ignored"}]) == ""
        assert module._extract_final_text_response(
            [
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ]
        ) == ""
        assert module._extract_final_text_response(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "line one"},
                        {"type": "tool_call", "name": "lookup"},
                        {"type": "text", "text": "line two"},
                    ],
                }
            ]
        ) == "line one\nline two"

    def test_message_parser_preprocessing_failure_returns_empty(self, monkeypatch):
        """A malformed message list degrades to an empty response."""
        module = self._evaluator_module

        def raise_preprocessing_error(messages):
            raise ValueError("invalid messages")

        monkeypatch.setattr(module, "_preprocess_messages", raise_preprocessing_error)
        assert module._parse_response_for_evaluation([{"role": "assistant", "content": "answer"}]) == ""

    @pytest.mark.parametrize("messages", ["not a list", []])
    def test_invalid_messages_raise_user_error(self, messages):
        """Invalid top-level messages use the SDK user-error convention."""
        with pytest.raises(EvaluationException) as exc_info:
            self._evaluator_module._response_from_messages(messages)

        assert exc_info.value.blame == ErrorBlame.USER_ERROR
        assert exc_info.value.category == ErrorCategory.INVALID_VALUE
        assert exc_info.value.target == ErrorTarget.EVALUATE


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

    def test_real_call_invalid_threshold_is_swallowed(self):
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
