# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable score regressions collected through each evaluator's CI test file."""

from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock
import wave

import aiohttp
import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import ErrorBlame, ErrorCategory, EvaluationException

from ..common.evaluator_mock_config import EVALUATOR_CONFIGS, EvaluatorCategory
from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_calls_evaluator_behavior_test import BaseToolCallEvaluatorBehaviorTest


def _make_evaluator(evaluator_class: Any) -> Any:
    """Construct the asset's evaluator without relying on ambient credentials."""
    return evaluator_class(
        model_config=AzureOpenAIModelConfiguration(
            azure_endpoint="https://example.openai.azure.com",
            azure_deployment="offline-score-test",
        ),
        credential=Mock(),
    )


class JudgeScoreHandlingTests:
    """Exercise text score parsing through each declared asset test class."""

    evaluator_class: Any
    evaluator_name: str

    @pytest.fixture(params=("turn", "messages"))
    def score_case(self, request, monkeypatch):
        """Replace the text judge while retaining the real evaluator call path."""
        monkeypatch.setattr(
            "socket.getaddrinfo",
            Mock(side_effect=AssertionError("Score tests must not access the network.")),
        )
        evaluator = _make_evaluator(self.evaluator_class)
        judge = AsyncMock()
        evaluator._flow = judge
        if hasattr(evaluator, "_multi_turn_flow"):
            evaluator._multi_turn_flow = judge
        return evaluator, judge, self._text_inputs(request.param)

    def _text_inputs(self, mode: str) -> Dict[str, Any]:
        """Use valid existing fixtures for turn and messages routing."""
        if mode == "messages":
            inputs = {
                "messages": deepcopy(
                    BaseEvaluatorBehaviorTest.VALID_QUERY + BaseEvaluatorBehaviorTest.VALID_RESPONSE
                ),
            }
        else:
            tool_aware = self.evaluator_name.startswith("tool_") or self.evaluator_name == "task_adherence"
            inputs = {
                "query": deepcopy(BaseEvaluatorBehaviorTest.VALID_QUERY)
                if tool_aware else BaseEvaluatorBehaviorTest.STRING_QUERY,
                "response": deepcopy(BaseEvaluatorBehaviorTest.VALID_RESPONSE)
                if tool_aware else BaseEvaluatorBehaviorTest.STRING_RESPONSE,
            }
        if self.evaluator_name.startswith("tool_") or self.evaluator_name == "task_adherence":
            inputs["tool_definitions"] = deepcopy(BaseToolCallEvaluatorBehaviorTest.VALID_TOOL_DEFINITIONS)
        if self.evaluator_name == "response_completeness":
            inputs["ground_truth"] = BaseEvaluatorBehaviorTest.STRING_RESPONSE
        return inputs

    @staticmethod
    def _evaluate(score_case, llm_output: Dict[str, Any]) -> Dict[str, Any]:
        """Inject judge output through the public evaluator call."""
        evaluator, judge, inputs = score_case
        judge.return_value = {
            "llm_output": llm_output,
            "input_token_count": 7,
            "output_token_count": 3,
            "total_token_count": 10,
            "model_id": "offline-judge",
        }
        return evaluator(**deepcopy(inputs))

    @pytest.mark.parametrize("status_fields", [
        pytest.param({"status": "completed"}, id="completed"),
        pytest.param({}, id="default-status"),
        pytest.param({"status": "error"}, id="judge-error"),
    ])
    def test_null_score_is_a_classified_execution_error(self, score_case, status_fields):
        """Non-skipped null scores must not escape as float(None) TypeErrors."""
        with pytest.raises(EvaluationException, match="null score") as exc_info:
            self._evaluate(score_case, {"score": None, "reason": "No grade available.", **status_fields})

        assert exc_info.value.blame == ErrorBlame.SYSTEM_ERROR
        assert exc_info.value.category == ErrorCategory.FAILED_EXECUTION
        score_case[1].assert_awaited_once()

    @pytest.mark.parametrize("outcome", ["fail", "pass", "fractional", "numeric-string"])
    def test_valid_scores_keep_quality_and_metadata(self, score_case, outcome):
        """Explicit outcomes ensure the fail and fractional controls really fail."""
        evaluator, judge, _ = score_case
        binary = EVALUATOR_CONFIGS[evaluator._result_key].category == EvaluatorCategory.BINARY
        scores = {
            "fail": 0 if binary else 1,
            "pass": 1 if binary else 5,
            "fractional": 0.5 if binary else 2.5,
            "numeric-string": "1" if binary else "5",
        }
        score = scores[outcome]
        result = self._evaluate(score_case, {
            "score": score,
            "status": "completed",
            "reason": "Evaluated.",
            "properties": {"judgeConfidence": "high"},
        })

        key = evaluator._result_key
        passed = outcome in ("pass", "numeric-string")
        assert result[key] == float(score)
        assert result[f"{key}_score"] == float(score)
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
        result = self._evaluate(score_case, {
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


class AudioJudgeScoreHandlingTests:
    """Exercise score validation through the real Realtime client."""

    evaluator_class: Any
    judge_properties: Dict[str, Any]

    @pytest.fixture
    def audio_case(self, tmp_path, monkeypatch):
        """Keep the Realtime client real and block unintended network access."""
        monkeypatch.setattr(
            "socket.getaddrinfo",
            Mock(side_effect=AssertionError("Audio score tests must not access the network.")),
        )
        evaluator = _make_evaluator(self.evaluator_class)
        audio_path = tmp_path / "call.wav"
        with wave.open(str(audio_path), "wb") as audio:
            audio.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(4_800))
        return evaluator, {"audio_path": str(audio_path)}

    def _mock_websocket(self, monkeypatch, payload: Dict[str, Any]) -> Mock:
        """Mock only authentication and WebSocket I/O, not response validation."""
        events = [
            {"type": "session.created"},
            {
                "type": "session.updated",
                "session": {
                    "model": "offline-judge",
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": None}},
                },
            },
            {"type": "input_audio_buffer.committed"},
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output": [{
                        "type": "function_call",
                        "name": "submit_evaluation",
                        "arguments": json.dumps(payload),
                    }],
                    "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                },
            },
        ]
        session = Mock(return_value=_RealtimeSession(events))
        monkeypatch.setattr(aiohttp, "ClientSession", session)
        return session

    def test_realtime_null_score_is_a_classified_execution_error(self, audio_case, monkeypatch):
        """The actual client must classify null scores before evaluator parsing."""
        evaluator, inputs = audio_case
        evaluator._realtime_client._headers = AsyncMock(return_value={})
        parser = Mock(wraps=evaluator._parse_prompty_output)
        evaluator._parse_prompty_output = parser
        session = self._mock_websocket(monkeypatch, {
            "score": None,
            "status": "completed",
            "reason": "The judge could not assign a score.",
            "properties": deepcopy(self.judge_properties),
        })

        with pytest.raises(EvaluationException, match="null score") as exc_info:
            evaluator(**inputs)

        assert exc_info.value.blame == ErrorBlame.SYSTEM_ERROR
        assert exc_info.value.category == ErrorCategory.FAILED_EXECUTION
        parser.assert_not_called()
        session.assert_called_once()

    @pytest.mark.parametrize("outcome", ["fail", "pass", "skipped"])
    def test_realtime_valid_and_skipped_scores_are_preserved(self, audio_case, monkeypatch, outcome):
        """Real client validation still accepts integer grades and explicit skips."""
        evaluator, inputs = audio_case
        evaluator._realtime_client._headers = AsyncMock(return_value={})
        binary = evaluator._result_key == "task_completion"
        score = (1 if binary else 5) if outcome == "pass" else (0 if binary else 1)
        if outcome == "skipped":
            score = None
        session = self._mock_websocket(monkeypatch, {
            "score": score,
            "status": "skipped" if outcome == "skipped" else "completed",
            "reason": "Evaluated.",
            "properties": None if outcome == "skipped" else deepcopy(self.judge_properties),
        })

        result = evaluator(**inputs)

        key = evaluator._result_key
        assert result[key] == score
        assert result[f"{key}_score"] == score
        assert result[f"{key}_result"] == ("not_applicable" if outcome == "skipped" else outcome)
        assert result[f"{key}_status"] == ("skipped" if outcome == "skipped" else "completed")
        expected_passed = None if outcome == "skipped" else outcome == "pass"
        assert result[f"{key}_passed"] is expected_passed
        if outcome != "skipped":
            assert result[f"{key}_total_tokens"] == 10
            assert result[f"{key}_model"] == "offline-judge"
        session.assert_called_once()


class _RealtimeSession:
    """Minimal transport double shared by all four real audio client paths."""

    def __init__(self, events: List[Dict[str, Any]]) -> None:
        """Retain an independent event queue for each WebSocket connection."""
        self._events = events

    async def __aenter__(self):
        """Enter the mocked client session."""
        return self

    async def __aexit__(self, *args):
        """Leave exceptions visible to the evaluator."""
        return False

    def ws_connect(self, *args, **kwargs):
        """Create a fresh socket without network or retry side effects."""
        socket = AsyncMock()
        socket.__aenter__.return_value = socket
        socket.receive.side_effect = [
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))
            for event in self._events
        ]
        return socket
