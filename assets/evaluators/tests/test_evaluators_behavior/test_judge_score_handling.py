# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Regression tests for unscored judge outputs across evaluator call paths."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import wave

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import ErrorBlame, ErrorCategory, EvaluationException

from ...builtin.coherence_audio.evaluator._coherence_audio import CoherenceAudioEvaluator
from ...builtin.customer_satisfaction_audio.evaluator._customer_satisfaction_audio import (
    CustomerSatisfactionAudioEvaluator,
)
from ...builtin.groundedness_audio.evaluator._groundedness_audio import GroundednessAudioEvaluator
from ...builtin.relevance.evaluator._relevance import RelevanceEvaluator
from ...builtin.response_completeness.evaluator._response_completeness import ResponseCompletenessEvaluator
from ...builtin.task_adherence.evaluator._task_adherence import TaskAdherenceEvaluator
from ...builtin.task_completion.evaluator._task_completion import TaskCompletionEvaluator
from ...builtin.task_completion_audio.evaluator._task_completion_audio import TaskCompletionAudioEvaluator
from ...builtin.tool_call_success.evaluator._tool_call_success import ToolCallSuccessEvaluator
from ...builtin.tool_output_utilization.evaluator._tool_output_utilization import ToolOutputUtilizationEvaluator
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest


_TEXT_EVALUATORS = [
    ("task_completion", TaskCompletionEvaluator),
    ("task_adherence", TaskAdherenceEvaluator),
    ("tool_call_success", ToolCallSuccessEvaluator),
    ("tool_output_utilization", ToolOutputUtilizationEvaluator),
    ("relevance", RelevanceEvaluator),
    ("response_completeness", ResponseCompletenessEvaluator),
]
_AUDIO_EVALUATORS = [
    ("coherence_audio", CoherenceAudioEvaluator),
    ("customer_satisfaction_audio", CustomerSatisfactionAudioEvaluator),
    ("groundedness_audio", GroundednessAudioEvaluator),
    ("task_completion_audio", TaskCompletionAudioEvaluator),
]
_SCORE_CASES = [
    pytest.param((name, evaluator_cls, mode), id=f"{name}-{mode}")
    for name, evaluator_cls in _TEXT_EVALUATORS
    for mode in ("turn", "messages")
] + [
    pytest.param((name, evaluator_cls, "audio"), id=name)
    for name, evaluator_cls in _AUDIO_EVALUATORS
]


@pytest.fixture(params=_SCORE_CASES)
def score_case(request, tmp_path, monkeypatch):
    """Build a real evaluator while replacing only its external judge call."""
    name, evaluator_cls, mode = request.param
    monkeypatch.setattr(
        "socket.getaddrinfo",
        Mock(side_effect=AssertionError("Score regression tests must not access the network.")),
    )
    evaluator = evaluator_cls(
        model_config=AzureOpenAIModelConfiguration(
            azure_endpoint="https://example.openai.azure.com",
            azure_deployment="offline-score-test",
        ),
        credential=Mock(),
    )
    judge = AsyncMock()
    for attr in ("_flow", "_multi_turn_flow", "_groundedness_flow"):
        if hasattr(evaluator, attr):
            setattr(evaluator, attr, judge)
    if hasattr(evaluator, "_ensure_query_prompty_loaded"):
        evaluator._ensure_query_prompty_loaded = Mock()

    if mode == "audio":
        audio_path = tmp_path / "call.wav"
        with wave.open(str(audio_path), "wb") as audio:
            audio.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(4_800))
        inputs = {"audio_path": str(audio_path)}
        evaluator._realtime_client.evaluate = judge
    elif mode == "messages":
        inputs = {
            "messages": deepcopy(
                BaseEvaluatorBehaviorTest.VALID_QUERY + BaseEvaluatorBehaviorTest.VALID_RESPONSE
            ),
        }
    else:
        tool_aware = name.startswith("tool_") or name == "task_adherence"
        inputs = {
            "query": deepcopy(BaseEvaluatorBehaviorTest.VALID_QUERY)
            if tool_aware else BaseEvaluatorBehaviorTest.STRING_QUERY,
            "response": deepcopy(BaseEvaluatorBehaviorTest.VALID_RESPONSE)
            if tool_aware else BaseEvaluatorBehaviorTest.STRING_RESPONSE,
        }
    if name.startswith("tool_") or name == "task_adherence":
        inputs["tool_definitions"] = deepcopy(BaseToolCallEvaluatorBehaviorTest.VALID_TOOL_DEFINITIONS)
    if name == "response_completeness":
        inputs["ground_truth"] = BaseEvaluatorBehaviorTest.STRING_RESPONSE
    return evaluator, judge, inputs


def _evaluate(score_case, llm_output):
    """Inject a judge response through the evaluator's normal public call."""
    evaluator, judge, inputs = score_case
    if hasattr(evaluator, "_realtime_client"):
        judge.return_value = SimpleNamespace(
            evaluation=llm_output,
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            status="completed",
            model="offline-judge",
        )
    else:
        judge.return_value = {
            "llm_output": llm_output,
            "input_token_count": 7,
            "output_token_count": 3,
            "total_token_count": 10,
            "model_id": "offline-judge",
        }
    return evaluator(**deepcopy(inputs))


@pytest.mark.unittest
class TestJudgeScoreHandling:
    """Reject null scores without mistaking judge failures for quality grades."""

    @pytest.mark.parametrize("status_fields", [
        pytest.param({"status": "completed"}, id="completed"),
        pytest.param({}, id="default-status"),
        pytest.param({"status": "error"}, id="judge-error"),
    ])
    def test_null_score_is_a_classified_execution_error(self, score_case, status_fields):
        """Non-skipped null scores must not escape as float(None) TypeErrors."""
        with pytest.raises(EvaluationException, match="null score") as exc_info:
            _evaluate(score_case, {"score": None, "reason": "No grade available.", **status_fields})

        assert exc_info.value.blame == ErrorBlame.SYSTEM_ERROR
        assert exc_info.value.category == ErrorCategory.FAILED_EXECUTION
        score_case[1].assert_awaited_once()

    @pytest.mark.parametrize("outcome", ["fail", "pass", "fractional", "numeric-string"])
    def test_valid_scores_keep_quality_and_metadata(self, score_case, outcome):
        """Keep valid zero, fractional and numeric-string scores unchanged."""
        evaluator, judge, _ = score_case
        binary = evaluator._threshold == 1
        scores = {
            "fail": 0 if binary else 1,
            "pass": 1 if binary else 5,
            "fractional": 0.5 if binary else 2.5,
            "numeric-string": "1" if binary else "5",
        }
        score = scores[outcome]
        result = _evaluate(score_case, {
            "score": score,
            "status": "completed",
            "reason": "Evaluated.",
            "properties": {"judgeConfidence": "high"},
        })

        key = evaluator._result_key
        expected_score = float(score)
        passed = expected_score >= evaluator._threshold
        assert result[key] == expected_score
        assert result[f"{key}_score"] == expected_score
        assert result[f"{key}_passed"] is passed
        assert result[f"{key}_result"] == ("pass" if passed else "fail")
        assert result[f"{key}_status"] == "completed"
        assert result[f"{key}_reason"] == "Evaluated."
        assert result[f"{key}_properties"]["judgeConfidence"] == "high"
        assert result[f"{key}_total_tokens"] == 10
        assert result[f"{key}_model"] == "offline-judge"
        judge.assert_awaited_once()

    def test_explicit_skip_with_null_score_remains_not_applicable(self, score_case):
        """A legitimate skip is not a judge-output error."""
        evaluator, judge, _ = score_case
        result = _evaluate(score_case, {
            "score": None,
            "status": "skipped",
            "reason": "No applicable response.",
            "properties": None,
        })

        key = evaluator._result_key
        assert result[key] is None
        assert result[f"{key}_score"] is None
        assert result[f"{key}_passed"] is None
        assert result[f"{key}_result"] == "not_applicable"
        assert result[f"{key}_status"] == "skipped"
        assert "No applicable response." in result[f"{key}_reason"]
        judge.assert_awaited_once()
