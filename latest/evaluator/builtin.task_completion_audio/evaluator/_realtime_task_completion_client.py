# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Azure OpenAI Realtime transport for Task Completion audio evaluation."""

import base64
import inspect
import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import aiohttp


class RealtimeTaskCompletionFormatError(ValueError):
    """Indicate a retryable function-call or JSON format failure."""


@dataclass(frozen=True)
class RealtimeTaskCompletionResponse:
    """Parsed Realtime evaluation and judge metadata."""

    evaluation: dict[str, Any]
    model: str
    status: str
    usage: dict[str, Any]


class RealtimeTaskCompletionClient:
    """Send one rendered Task Completion prompt and one whole PCM recording."""

    _SAMPLE_RATE = 24_000
    _CHUNK_BYTES = _SAMPLE_RATE * 2
    _EVENT_TIMEOUT_SECONDS = 120
    _SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(
        self,
        model_config: Any,
        credential: Any,
        logger: Any,
        *,
        max_output_tokens: int = 3_000,
    ):
        """Configure the Realtime endpoint from the evaluator model config."""
        self._endpoint = str(self._config_value(model_config, "azure_endpoint") or "").rstrip("/")
        self._model = str(self._config_value(model_config, "azure_deployment") or "")
        self._api_key = self._config_value(model_config, "api_key")
        if not self._endpoint or not self._model:
            raise ValueError("Realtime Task Completion requires azure_endpoint and azure_deployment.")
        if not self._api_key and credential is None:
            raise ValueError("Realtime Task Completion requires credential or api_key.")
        self._credential = credential
        self._logger = logger
        self._max_output_tokens = max_output_tokens

    async def evaluate(
        self,
        pcm: bytes,
        instructions: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> RealtimeTaskCompletionResponse:
        """Evaluate one full conversation with exactly one audio commit."""
        endpoint = self._endpoint.replace("https://", "wss://", 1)
        url = f"{endpoint}/openai/v1/realtime?{urlencode({'model': self._model})}"
        timeout = aiohttp.ClientTimeout(total=None, connect=30, sock_read=self._EVENT_TIMEOUT_SECONDS)
        headers = await self._headers()

        self._logger.info("Calling Realtime Task Completion judge model %s.", self._model)
        for attempt in range(2):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.ws_connect(url, headers=headers) as socket:
                    await self._receive(socket, "session.created")
                    await socket.send_json(
                        {"type": "session.update", "session": self._build_session(instructions, schema)}
                    )
                    session_event = await self._receive(socket, "session.updated")
                    self._validate_session(session_event["session"])
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
                    await self._send_audio(socket, pcm)
                    await socket.send_json(
                        {"type": "response.create", "response": {"output_modalities": ["text"]}}
                    )
                    response = (await self._receive(socket, "response.done"))["response"]
            try:
                return self._parse_response(response, session_event["session"])
            except (json.JSONDecodeError, RealtimeTaskCompletionFormatError):
                if attempt:
                    raise
                self._logger.warning("Realtime returned invalid Task Completion output; retrying once.")
        raise RuntimeError("Unreachable Realtime Task Completion retry state.")

    async def _headers(self) -> dict[str, str]:
        """Build API-key or Entra authentication headers."""
        if self._api_key:
            return {"api-key": str(self._api_key)}
        token = self._credential.get_token(self._SCOPE)
        if inspect.isawaitable(token):
            token = await token
        return {"Authorization": f"Bearer {token.token}"}

    def _build_session(self, instructions: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Build the text-only, manual-turn Realtime session."""
        session = {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["text"],
            "max_output_tokens": self._max_output_tokens,
            "truncation": "disabled",
            "audio": {
                "input": {
                    "turn_detection": None,
                    "transcription": None,
                    "noise_reduction": None,
                    "format": {"type": "audio/pcm", "rate": self._SAMPLE_RATE},
                }
            },
            "tools": [
                {
                    "type": "function",
                    "name": "submit_evaluation",
                    "description": (
                        "Return the evaluation as function arguments. "
                        "No external action is executed."
                    ),
                    "parameters": schema,
                }
            ],
            "tool_choice": "required",
        }
        return session

    @staticmethod
    def _validate_session(session: dict[str, Any]) -> None:
        """Ensure Realtime applied the required audio contract."""
        audio_input = session.get("audio", {}).get("input", {})
        if session.get("output_modalities") != ["text"] or audio_input.get("turn_detection") is not None:
            raise ValueError("Realtime did not apply text-only manual-turn configuration.")

    @classmethod
    async def _send_audio(cls, socket: aiohttp.ClientWebSocketResponse, pcm: bytes) -> None:
        """Append one-second PCM chunks and commit the whole call once."""
        for offset in range(0, len(pcm), cls._CHUNK_BYTES):
            await socket.send_json(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm[offset:offset + cls._CHUNK_BYTES]).decode(),
                }
            )
        await socket.send_json({"type": "input_audio_buffer.commit"})
        await cls._receive(socket, "input_audio_buffer.committed")

    @classmethod
    async def _receive(
        cls, socket: aiohttp.ClientWebSocketResponse, event_type: str
    ) -> dict[str, Any]:
        """Receive JSON events until the requested event arrives."""
        while True:
            message = await socket.receive(timeout=cls._EVENT_TIMEOUT_SECONDS)
            if message.type != aiohttp.WSMsgType.TEXT:
                raise ValueError(f"Realtime WebSocket closed before {event_type}: {message.type.name}")
            event = json.loads(message.data)
            if event.get("type") == "error":
                raise ValueError(f"Realtime error: {event.get('error')}")
            if event.get("type") == event_type:
                return event

    @staticmethod
    def _parse_response(
        response: dict[str, Any], session: dict[str, Any]
    ) -> RealtimeTaskCompletionResponse:
        """Parse the schema-enforced submit_evaluation function call."""
        if response.get("status") != "completed":
            raise ValueError(f"Realtime response was not completed: {response.get('status_details')}")
        calls = [
            item
            for item in response.get("output", [])
            if item.get("type") == "function_call" and item.get("name") == "submit_evaluation"
        ]
        if len(calls) != 1:
            raise RealtimeTaskCompletionFormatError(
                "Realtime must return exactly one submit_evaluation call."
            )
        evaluation = json.loads(calls[0]["arguments"])
        if not isinstance(evaluation, dict) or not {
            "reason",
            "properties",
            "score",
            "status",
        }.issubset(evaluation):
            raise RealtimeTaskCompletionFormatError(
                "Realtime Task Completion output is missing fields."
            )
        reason = evaluation["reason"]
        status = evaluation["status"]
        score = evaluation["score"]
        properties = evaluation["properties"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Evaluation requires a nonempty reason.")
        if status == "skipped":
            if score is not None or properties is not None:
                raise ValueError("Skipped evaluation must have null score/properties.")
        elif status == "completed":
            if (
                type(score) is not int
                or score not in (0, 1)
                or not isinstance(properties, dict)
                or not isinstance(properties.get("user_requests"), list)
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in properties["user_requests"]
                )
                or any(
                    not isinstance(properties.get(key), str) or not properties[key].strip()
                    for key in ("delivered_outcome", "completion_gaps")
                )
            ):
                raise ValueError("Invalid Task Completion score/properties.")
        else:
            raise ValueError(f"Unknown Task Completion status: {status}")
        model = str(session.get("model") or "").strip()
        if not model:
            raise ValueError("Realtime session did not include a resolved model identifier.")
        return RealtimeTaskCompletionResponse(
            evaluation=evaluation,
            model=model,
            status=str(response["status"]),
            usage=response.get("usage") or {},
        )

    @staticmethod
    def _config_value(model_config: Any, name: str) -> Any:
        """Read a model-config field from a mapping or SDK model."""
        if isinstance(model_config, dict):
            return model_config.get(name)
        return getattr(model_config, name, None)
