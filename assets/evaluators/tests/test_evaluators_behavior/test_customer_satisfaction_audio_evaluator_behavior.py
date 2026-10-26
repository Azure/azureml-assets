# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral and unit tests for Customer Satisfaction Audio Evaluator."""

import asyncio
import json
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest
from .base_judge_score_handling_test import AudioJudgeScoreHandlingTests
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from ...builtin.customer_satisfaction_audio.evaluator._customer_satisfaction_audio import (
    CustomerSatisfactionAudioEvaluator,
)
from ...builtin.customer_satisfaction_audio.evaluator._realtime_customer_satisfaction_client import (
    RealtimeCustomerSatisfactionClient,
    RealtimeCustomerSatisfactionFormatError,
    RealtimeCustomerSatisfactionResponse,
)


def _write_wav(
    path: Path,
    frames: bytes = bytes(4_800),
    rate: int = 24_000,
    channels: int = 1,
    width: int = 2,
) -> Path:
    """Write a test WAV file."""
    with wave.open(str(path), "wb") as audio:
        audio.setparams((channels, width, rate, 0, "NONE", "not compressed"))
        audio.writeframes(frames)
    return path


def _make_evaluator(**kwargs) -> CustomerSatisfactionAudioEvaluator:
    """Create a CustomerSatisfactionAudioEvaluator with test model configuration."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint="https://example.openai.azure.com",
        azure_deployment="gpt-realtime-2",
    )
    return CustomerSatisfactionAudioEvaluator(model_config=model_config, credential=Mock(), **kwargs)


@pytest.mark.unittest
class TestCustomerSatisfactionAudioJudgeScores(AudioJudgeScoreHandlingTests):
    """Exercise score validation at the real Realtime client boundary."""

    evaluator_class = CustomerSatisfactionAudioEvaluator
    evaluator_name = "customer_satisfaction_audio"
    judge_properties = {
        "helpfulness": "Helpful.",
        "completeness": "Complete.",
        "tone": "Polite.",
    }


@pytest.mark.unittest
class TestCustomerSatisfactionAudioEvaluatorBehavior:
    """Tests for CustomerSatisfactionAudioEvaluator and RealtimeCustomerSatisfactionClient."""

    def test_realtime_audio_reuses_multiturn_contract(self, tmp_path: Path):
        """Realtime audio keeps the text evaluator's prompt, parser, threshold, and output keys."""
        evaluator = _make_evaluator(evaluation_level="conversation", threshold=4)
        evaluator._realtime_client.evaluate = AsyncMock(
            return_value=RealtimeCustomerSatisfactionResponse(
                evaluation={
                    "reason": "Order cancelled smoothly and all questions answered.",
                    "properties": {
                        "helpfulness": "Addressed all user needs.",
                        "completeness": "All requests resolved.",
                        "tone": "Polite and clear.",
                    },
                    "score": 5,
                    "status": "completed",
                },
                model="gpt-realtime-2",
                status="completed",
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        )
        audio_path = _write_wav(tmp_path / "call.wav")
        messages = [
            {"role": "developer", "content": "Follow the cancellation policy."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "cancel_order",
                        "arguments": {"order_id": "88412"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "tool_result", "tool_result": {"status": "cancelled"}}],
            },
        ]
        result = evaluator(
            messages=messages,
            audio_path=str(audio_path),
        )

        assert result["customer_satisfaction"] == 5.0
        assert result["customer_satisfaction_score"] == 5.0
        assert result["customer_satisfaction_passed"] is True
        assert result["customer_satisfaction_result"] == "pass"
        assert result["customer_satisfaction_threshold"] == 4
        assert result["customer_satisfaction_model"] == "gpt-realtime-2"
        assert result["customer_satisfaction_properties"]["helpfulness"] == "Addressed all user needs."
        call = evaluator._realtime_client.evaluate.await_args
        assert call.args[1] == evaluator._audio_system_prompt
        prompt = call.args[2]
        assert "Follow the cancellation policy." in prompt
        assert "[TOOL_CALL] cancel_order" in prompt
        assert call.args[3] == evaluator._RESULT_SCHEMA

        auto_evaluator = _make_evaluator()
        auto_evaluator._realtime_client.evaluate = AsyncMock(
            return_value=RealtimeCustomerSatisfactionResponse(
                evaluation={
                    "reason": "User request left unresolved.",
                    "properties": {
                        "helpfulness": "Unhelpful generic answers.",
                        "completeness": "Request not resolved.",
                        "tone": "Evasive.",
                    },
                    "score": 2,
                    "status": "completed",
                },
                model="gpt-realtime-2",
                status="completed",
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        )
        fail_result = auto_evaluator(messages=messages, audio_path=str(audio_path))
        assert fail_result["customer_satisfaction"] == 2.0
        assert fail_result["customer_satisfaction_passed"] is False
        assert fail_result["customer_satisfaction_result"] == "fail"
        assert fail_result["customer_satisfaction_threshold"] == 3
        assert not auto_evaluator._should_use_conversation_level({})

        empty_msg_result = auto_evaluator(messages=[], audio_path=str(audio_path))
        assert empty_msg_result["customer_satisfaction"] == 2.0
        empty_prompt = auto_evaluator._realtime_client.evaluate.await_args.args[2]
        assert "[AUDIO: The complete call recording supplies the spoken user and agent turns.]" in empty_prompt

        omitted_msg_result = auto_evaluator(audio_path=str(audio_path))
        assert omitted_msg_result["customer_satisfaction"] == 2.0

    def test_evaluator_skipped_and_validation_errors(self, tmp_path: Path):
        """Verify skipped output, invalid init parameters, and input validation errors."""
        with pytest.raises(ValueError, match="judge_transport"):
            _make_evaluator(judge_transport="text")
        with pytest.raises(ValueError, match="max_output_tokens"):
            _make_evaluator(max_output_tokens=0)
        with pytest.raises(ValueError, match="conversation-level"):
            _make_evaluator(evaluation_level="turn")

        evaluator = _make_evaluator()
        with pytest.raises(EvaluationException, match="messages must be a list"):
            asyncio.run(evaluator._real_call(messages="not-a-list", audio_path="call.wav"))
        with pytest.raises(EvaluationException, match="audio_path"):
            evaluator(messages=[{"role": "user", "content": "Hi"}])

        stereo_16k = _write_wav(tmp_path / "stereo_16k.wav", frames=bytes(6_400), rate=16_000, channels=2)
        converted_pcm = evaluator._read_pcm(stereo_16k)
        assert len(converted_pcm) == 4_800

        bad_width = _write_wav(tmp_path / "bad_width.wav", frames=bytes(2_400), width=1)
        with pytest.raises(ValueError, match="uncompressed PCM16 WAV"):
            evaluator._read_pcm(bad_width)

        empty_wav = _write_wav(tmp_path / "empty.wav", frames=b"")
        with pytest.raises(ValueError, match="must not be empty"):
            evaluator._read_pcm(empty_wav)

        with pytest.raises(EvaluationException, match="invalid output"):
            evaluator._parse_prompty_output({"llm_output": "bad"})

        skipped = evaluator._parse_prompty_output(
            {"llm_output": {"reason": "Empty conversation", "properties": None, "score": None, "status": "skipped"}}
        )
        assert skipped["customer_satisfaction"] is None
        assert skipped["customer_satisfaction_passed"] is None
        assert skipped["customer_satisfaction_result"] == "not_applicable"
        assert skipped["customer_satisfaction_status"] == "skipped"

        built = evaluator._build_result(
            score=4,
            result="pass",
            reason="ok",
            status="completed",
            properties={},
        )
        assert built["customer_satisfaction_passed"] is True

    @pytest.mark.asyncio
    async def test_realtime_client_end_to_end_and_retry(self, monkeypatch):
        """Verify WebSocket session flow, retry on format error, and auth headers."""
        with pytest.raises(ValueError, match="azure_endpoint"):
            RealtimeCustomerSatisfactionClient({}, Mock(), Mock())
        with pytest.raises(ValueError, match="credential or api_key"):
            RealtimeCustomerSatisfactionClient(
                {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "gpt-realtime-2"},
                None,
                Mock(),
            )

        key_client = RealtimeCustomerSatisfactionClient(
            SimpleNamespace(
                azure_endpoint="https://example.openai.azure.com",
                azure_deployment="gpt-realtime-2",
                api_key="secret",
            ),
            None,
            Mock(),
        )
        assert await key_client._headers() == {"api-key": "secret"}

        async def _token_coro():
            return SimpleNamespace(token="tok")

        cred = Mock()
        cred.get_token = Mock(side_effect=[SimpleNamespace(token="sync_tok"), _token_coro()])
        client = RealtimeCustomerSatisfactionClient(
            {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "gpt-realtime-2"},
            cred,
            Mock(),
            max_output_tokens=123,
        )
        assert (await client._headers())["Authorization"].endswith("sync_tok")
        assert (await client._headers())["Authorization"].endswith("tok")
        cred.get_token = Mock(return_value=SimpleNamespace(token="sync_tok"))

        valid_args = json.dumps(
            {
                "reason": "Very satisfied.",
                "properties": {
                    "helpfulness": "Helpful",
                    "completeness": "Complete",
                    "tone": "Polite",
                },
                "score": 5,
                "status": "completed",
            }
        )
        events_attempt_1 = [
            {"type": "session.created"},
            {
                "type": "session.updated",
                "session": {
                    "model": "gpt-realtime-2",
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": None}},
                },
            },
            {"type": "input_audio_buffer.committed"},
            {"type": "response.done", "response": {"status": "completed", "output": []}},
        ]
        events_attempt_2 = [
            {"type": "session.created"},
            {
                "type": "session.updated",
                "session": {
                    "model": "gpt-realtime-2-2025",
                    "output_modalities": ["text"],
                    "audio": {"input": {"turn_detection": None}},
                },
            },
            {"type": "noise.ignored"},
            {"type": "input_audio_buffer.committed"},
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output": [{"type": "function_call", "name": "submit_evaluation", "arguments": valid_args}],
                    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                },
            },
        ]
        attempts = [events_attempt_1, events_attempt_2]

        class _FakeSocket:
            def __init__(self, queue):
                self._queue = list(queue)
                self.sent = []

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def send_json(self, payload):
                if payload["type"] == "session.update":
                    assert payload["session"]["tool_choice"] == "required"
                self.sent.append(payload)

            async def receive(self, timeout=None):
                item = self._queue.pop(0)
                return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(item))

        class _FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def ws_connect(self, url, headers=None):
                return _FakeSocket(attempts.pop(0))

        monkeypatch.setattr(aiohttp, "ClientSession", _FakeSession)
        res = await client.evaluate(
            bytes(RealtimeCustomerSatisfactionClient._CHUNK_BYTES + 2),
            "sys",
            "usr",
            {"type": "object"},
        )
        assert res.model == "gpt-realtime-2-2025"
        assert res.evaluation["score"] == 5

        attempts.extend([events_attempt_1, events_attempt_1])
        with pytest.raises(RealtimeCustomerSatisfactionFormatError):
            await client.evaluate(b"\x00\x00", "sys", "usr", {"type": "object"})

    @pytest.mark.asyncio
    async def test_realtime_client_validation_and_receive_errors(self):
        """Verify session validation, receive errors, and response parser edge cases."""
        client = RealtimeCustomerSatisfactionClient(
            {"azure_endpoint": "https://example.openai.azure.com", "azure_deployment": "gpt-realtime-2"},
            Mock(),
            Mock(),
        )
        with pytest.raises(ValueError, match="manual-turn"):
            client._validate_session({"output_modalities": ["audio"], "audio": {"input": {"turn_detection": None}}})

        closed_socket = AsyncMock()
        closed_socket.receive = AsyncMock(return_value=SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=""))
        with pytest.raises(ValueError, match="WebSocket closed"):
            await RealtimeCustomerSatisfactionClient._receive(closed_socket, "session.created")

        error_socket = AsyncMock()
        error_socket.receive = AsyncMock(
            return_value=SimpleNamespace(
                type=aiohttp.WSMsgType.TEXT,
                data=json.dumps({"type": "error", "error": {"message": "boom"}}),
            )
        )
        with pytest.raises(ValueError, match="Realtime error"):
            await RealtimeCustomerSatisfactionClient._receive(error_socket, "session.created")

        with pytest.raises(ValueError, match="not completed"):
            RealtimeCustomerSatisfactionClient._parse_response({"status": "failed"}, {"model": "gpt-realtime-2"})

        with pytest.raises(RealtimeCustomerSatisfactionFormatError, match="Invalid JSON"):
            RealtimeCustomerSatisfactionClient._parse_response(
                {
                    "status": "completed",
                    "output": [{"type": "function_call", "name": "submit_evaluation", "arguments": "{bad"}],
                },
                {"model": "gpt-realtime-2"},
            )

        def _call_resp(payload: dict, model: str = "gpt-realtime-2"):
            return RealtimeCustomerSatisfactionClient._parse_response(
                {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "name": "submit_evaluation",
                            "arguments": json.dumps(payload),
                        }
                    ],
                },
                {"model": model},
            )

        with pytest.raises(RealtimeCustomerSatisfactionFormatError, match="missing fields"):
            _call_resp({"reason": "only"})
        with pytest.raises(ValueError, match="nonempty reason"):
            _call_resp({"reason": " ", "properties": None, "score": None, "status": "skipped"})
        with pytest.raises(ValueError, match="null score/properties"):
            _call_resp({"reason": "ok", "properties": None, "score": 3, "status": "skipped"})
        skipped_resp = _call_resp({"reason": "ok", "properties": None, "score": None, "status": "skipped"})
        assert skipped_resp.evaluation["status"] == "skipped"
        with pytest.raises(ValueError, match="Invalid Customer Satisfaction score/properties"):
            _call_resp({"reason": "ok", "properties": {}, "score": 6, "status": "completed"})
        with pytest.raises(ValueError, match="Invalid Customer Satisfaction properties"):
            _call_resp({"reason": "ok", "properties": {"helpfulness": "a"}, "score": 4, "status": "completed"})
        with pytest.raises(ValueError, match="Unknown Customer Satisfaction status"):
            _call_resp({"reason": "ok", "properties": None, "score": None, "status": "other"})
        with pytest.raises(ValueError, match="resolved model identifier"):
            _call_resp(
                {
                    "reason": "ok",
                    "properties": {
                        "helpfulness": "a",
                        "completeness": "b",
                        "tone": "c",
                    },
                    "score": 4,
                    "status": "completed",
                },
                model="",
            )
