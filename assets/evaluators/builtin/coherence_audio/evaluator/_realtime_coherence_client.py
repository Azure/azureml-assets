# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Direct Azure OpenAI Realtime WebSocket client for audio Coherence evaluation."""

import asyncio
import base64
import inspect
import json
from dataclasses import dataclass
from typing import Any, Dict
from urllib.parse import quote

import aiohttp


class RealtimeCoherenceFormatError(ValueError):
    """Raised when the Realtime judge fails to return a valid submit_evaluation tool call."""


@dataclass(frozen=True)
class RealtimeCoherenceResponse:
    """Structured response from the Realtime Coherence judge."""

    evaluation: Dict[str, Any]
    model: str
    status: str
    usage: Dict[str, Any]


class RealtimeCoherenceClient:
    """Evaluate a call WAV plus conversation trace with Azure OpenAI Realtime over WebSocket."""

    _API_VERSION = "2025-08-28"
    _TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
    _CHUNK_BYTES = 240_000
    _TIMEOUT_SECONDS = 180.0
    _TOOL_NAME = "submit_evaluation"

    def __init__(
        self,
        model_config: Any,
        credential: Any,
        logger: Any,
        *,
        max_output_tokens: int = 4096,
    ) -> None:
        """Initialize the Realtime Coherence client."""
        self._endpoint = self._read_config(model_config, "azure_endpoint").rstrip("/")
        self._deployment = self._read_config(model_config, "azure_deployment")
        self._api_key = self._read_config(model_config, "api_key")
        self._credential = credential
        self._logger = logger
        self._max_output_tokens = max_output_tokens
        if not self._endpoint or not self._deployment:
            raise ValueError("model_config must include azure_endpoint and azure_deployment.")
        if not self._api_key and self._credential is None:
            raise ValueError("Realtime audio evaluation requires credential or api_key.")

    async def evaluate(
        self,
        pcm16_audio: bytes,
        system_prompt: str,
        user_prompt: str,
        result_schema: Dict[str, Any],
    ) -> RealtimeCoherenceResponse:
        """Run one Realtime evaluation call, retrying once on malformed tool output."""
        for attempt in range(2):
            try:
                return await self._evaluate_once(
                    pcm16_audio,
                    system_prompt,
                    user_prompt,
                    result_schema,
                )
            except RealtimeCoherenceFormatError:
                if attempt == 1:
                    raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    async def _evaluate_once(
        self,
        pcm16_audio: bytes,
        system_prompt: str,
        user_prompt: str,
        result_schema: Dict[str, Any],
    ) -> RealtimeCoherenceResponse:
        headers = await self._headers()
        timeout = aiohttp.ClientTimeout(total=self._TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.ws_connect(self._ws_url(), headers=headers) as socket:
                await self._receive(socket, "session.created")
                await socket.send_json(self._session_update(system_prompt, result_schema))
                updated = (await self._receive(socket, "session.updated")).get("session") or {}
                self._validate_session(updated)
                for offset in range(0, len(pcm16_audio), self._CHUNK_BYTES):
                    chunk = base64.b64encode(pcm16_audio[offset:offset + self._CHUNK_BYTES]).decode("ascii")
                    await socket.send_json({"type": "input_audio_buffer.append", "audio": chunk})
                await socket.send_json({"type": "input_audio_buffer.commit"})
                await self._receive(socket, "input_audio_buffer.committed")
                await socket.send_json(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": user_prompt}],
                        },
                    }
                )
                await socket.send_json({"type": "response.create"})
                done = await self._receive(socket, "response.done")
                return self._parse_response(done.get("response") or {}, updated)

    def _session_update(
        self,
        system_prompt: str,
        result_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": system_prompt,
                "output_modalities": ["text"],
                "max_output_tokens": self._max_output_tokens,
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": 24000},
                        "turn_detection": None,
                    }
                },
                "tools": [
                    {
                        "type": "function",
                        "name": self._TOOL_NAME,
                        "description": "Submit the final Coherence evaluation JSON.",
                        "parameters": result_schema,
                    }
                ],
                "tool_choice": "required",
            },
        }

    def _validate_session(self, session: Dict[str, Any]) -> None:
        audio_input = (session.get("audio") or {}).get("input") or {}
        if session.get("output_modalities") != ["text"] or audio_input.get("turn_detection") is not None:
            raise ValueError(f"Realtime session did not apply manual-turn text config: {session}")

    async def _headers(self) -> Dict[str, str]:
        if self._api_key:
            return {"api-key": self._api_key}
        token_result = self._credential.get_token(self._TOKEN_SCOPE)
        if inspect.isawaitable(token_result):
            token_result = await token_result
        else:
            token_result = await asyncio.to_thread(lambda: token_result)
        return {"Authorization": f"Bearer {token_result.token}"}

    def _ws_url(self) -> str:
        host = self._endpoint.removeprefix("https://").removeprefix("http://")
        deployment = quote(self._deployment, safe="")
        return f"wss://{host}/openai/v1/realtime?model={deployment}&api-version={self._API_VERSION}"

    @staticmethod
    def _read_config(model_config: Any, key: str) -> str:
        if isinstance(model_config, dict):
            return str(model_config.get(key) or "")
        return str(getattr(model_config, key, "") or "")

    @classmethod
    async def _receive(cls, socket: Any, expected_type: str) -> Dict[str, Any]:
        while True:
            message = await socket.receive(timeout=cls._TIMEOUT_SECONDS)
            if message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                raise ValueError(f"Realtime WebSocket closed while waiting for {expected_type}: {message.type}")
            if message.type != aiohttp.WSMsgType.TEXT:
                continue  # pragma: no cover
            event = json.loads(message.data)
            if event.get("type") == "error":
                raise ValueError(f"Realtime error waiting for {expected_type}: {event.get('error')}")
            if event.get("type") == expected_type:
                return event

    @classmethod
    def _parse_response(
        cls,
        response: Dict[str, Any],
        session: Dict[str, Any],
    ) -> RealtimeCoherenceResponse:
        status = str(response.get("status") or "")
        if status != "completed":
            raise ValueError(f"Realtime response was not completed: {response}")
        calls = [
            item
            for item in (response.get("output") or [])
            if item.get("type") == "function_call" and item.get("name") == cls._TOOL_NAME
        ]
        if len(calls) != 1:
            raise RealtimeCoherenceFormatError(
                f"Expected one {cls._TOOL_NAME} call, got {len(calls)}: {response}"
            )
        try:
            payload = json.loads(calls[0].get("arguments") or "")
        except json.JSONDecodeError as exc:
            raise RealtimeCoherenceFormatError(f"Invalid JSON in {cls._TOOL_NAME}: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"reason", "properties", "score", "status"}:
            raise RealtimeCoherenceFormatError(f"{cls._TOOL_NAME} missing fields: {payload}")
        reason = payload.get("reason")
        eval_status = payload.get("status")
        score = payload.get("score")
        props = payload.get("properties")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Coherence evaluation requires nonempty reason: {payload}")
        if eval_status == "skipped":
            if score is not None or (props is not None and not isinstance(props, dict)):
                raise ValueError(f"Skipped Coherence evaluation must have null score/properties: {payload}")
        elif eval_status == "completed":
            if type(score) is not int or not (1 <= score <= 5) or not isinstance(props, dict):
                raise ValueError(f"Invalid Coherence score/properties: {payload}")
            if (
                set(props) != {"gating_summary", "conversation_flow_summary", "agent_coherence_issues"}
                or not isinstance(props.get("gating_summary"), str)
                or not props["gating_summary"].strip()
                or not isinstance(props.get("conversation_flow_summary"), str)
                or not props["conversation_flow_summary"].strip()
                or not isinstance(props.get("agent_coherence_issues"), str)
                or not props["agent_coherence_issues"].strip()
            ):
                raise ValueError(f"Invalid Coherence properties: {payload}")
        else:
            raise ValueError(f"Unknown Coherence status: {payload}")
        model = str(session.get("model") or response.get("model") or "")
        if not model:
            raise ValueError("Realtime response did not include a resolved model identifier.")
        return RealtimeCoherenceResponse(
            evaluation=payload,
            model=model,
            status=status,
            usage=dict(response.get("usage") or {}),
        )
