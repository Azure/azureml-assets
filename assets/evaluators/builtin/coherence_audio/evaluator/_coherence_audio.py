# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import asyncio
import json
import logging
import math
import os
import wave
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import resample_poly
from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import (
    EvaluationException,
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
)
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    validate_model_config,
)
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
    ToolDefinitionsValidator,
)

try:
    from ._realtime_coherence_client import RealtimeCoherenceClient
except ImportError:  # pragma: no cover
    from _realtime_coherence_client import (  # type: ignore[no-redef]
        RealtimeCoherenceClient,
    )

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols.
# ---------------------------------------------------------------------------

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import (  # noqa: F401
        _is_intermediate_response,
        _preprocess_messages,
    )
except ImportError:  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _is_intermediate_response,
        _preprocess_messages,
    )

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )
except ImportError:  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._evaluators._common._validators import MessageRole
except ImportError:  # pragma: no cover
    class MessageRole(str, Enum):
        """Valid message roles in conversations."""

        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"
        TOOL = "tool"
        DEVELOPER = "developer"

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.constants import EvaluationLevel
    from azure.ai.evaluation._common.utils import (
        _resolve_evaluation_level,
        _wrap_string_messages,
        _merge_query_response_messages,
        _split_messages_at_latest_user,
        serialize_messages,
    )
    from azure.ai.evaluation._evaluators._common._validators import MessagesOrQueryResponseInputValidator
except ImportError:  # pragma: no cover
    from azure.ai.evaluation._common.utils import (
        _extract_text_from_content,
        _get_agent_response,
        _pretty_format_conversation_history,
    )

    class EvaluationLevel(str, Enum):
        """Supported evaluation levels for multi-turn evaluators."""

        CONVERSATION = "conversation"
        TURN = "turn"

    def _merge_query_response_messages(query: List[dict], response: List[dict]) -> List[dict]:
        return [*query, *response]

    def _split_messages_at_latest_user(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
        latest_user_index = max(
            (i for i, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        if latest_user_index == -1:
            raise EvaluationException(
                message="messages must contain at least one message with role 'user'.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            )
        return messages[: latest_user_index + 1], messages[latest_user_index + 1:]

    def _wrap_string_messages(query: str, response: str) -> Tuple[List[dict], List[dict]]:
        return (
            [{"role": "user", "content": [{"type": "text", "text": query}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
        )

    def _resolve_evaluation_level(
        evaluation_level: Optional[Union[EvaluationLevel, str]],
        error_target: ErrorTarget,
    ) -> Optional[EvaluationLevel]:
        valid = [level.value for level in EvaluationLevel]
        if evaluation_level is None or evaluation_level == "":
            return None
        if isinstance(evaluation_level, EvaluationLevel):
            return evaluation_level
        if isinstance(evaluation_level, str):
            try:
                return EvaluationLevel(evaluation_level)
            except ValueError:
                raise EvaluationException(
                    message=f"Invalid evaluation_level '{evaluation_level}'. Must be one of: {valid}.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=error_target,
                )
        raise EvaluationException(
            message=f"Invalid evaluation_level '{evaluation_level}'. Must be one of: {valid}.",
            blame=ErrorBlame.USER_ERROR,
            category=ErrorCategory.INVALID_VALUE,
            target=error_target,
        )

    def serialize_messages(messages):
        """Serialize a list of chat messages into a labeled text transcript."""
        if not messages:
            return ""

        all_user_queries = []
        all_agent_responses = []
        cur_user_query = []
        cur_agent_response = []
        system_message = None

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if not role:
                continue

            normalized = msg
            if role == MessageRole.ASSISTANT and isinstance(msg.get("content"), str):
                normalized = {**msg, "content": [{"type": "text", "text": msg["content"]}]}

            if role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
                content = msg.get("content", "")
                if isinstance(content, list):
                    system_message = "\n".join(_extract_text_from_content(content))
                else:
                    system_message = content
            elif role == MessageRole.USER and "content" in msg:
                if cur_agent_response:
                    formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
                    all_agent_responses.append([formatted])
                    cur_agent_response = []
                content = msg["content"]
                text_in_msg = [content] if isinstance(content, str) else _extract_text_from_content(content)
                if text_in_msg:
                    cur_user_query.append(text_in_msg)
            elif role in (MessageRole.ASSISTANT, MessageRole.TOOL):
                if cur_user_query:
                    all_user_queries.append(cur_user_query)
                    cur_user_query = []
                cur_agent_response.append(normalized)

        if cur_user_query:
            all_user_queries.append(cur_user_query)
        if cur_agent_response:
            formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
            all_agent_responses.append([formatted])

        conversation_history: Dict = {
            "user_queries": all_user_queries,
            "agent_responses": all_agent_responses[: len(all_user_queries) - 1] if all_user_queries else [],
        }
        if system_message:
            conversation_history["system_message"] = system_message

        result = _pretty_format_conversation_history(conversation_history)
        start = max(len(all_user_queries) - 1, 0)
        for i, agent_response in enumerate(all_agent_responses[start:], start=start):
            result += f"Agent turn {i + 1}:\n"
            for msg_text in agent_response:
                if isinstance(msg_text, list):
                    for submsg in msg_text:
                        result += "  " + "\n  ".join(submsg.split("\n")) + "\n"
                else:
                    result += "  " + "\n  ".join(msg_text.split("\n")) + "\n"
            result += "\n"

        return result.rstrip("\n")

    class MessagesOrQueryResponseInputValidator(ToolDefinitionsValidator):
        """Validator that supports both ``query``/``response`` and ``messages`` inputs."""

        enforce_tool_definitions: bool = False

        def __init__(
            self,
            error_target: ErrorTarget,
            requires_query: bool = True,
            optional_tool_definitions: bool = True,
            check_for_unsupported_tools: bool = False,
            *,
            enforce_tool_definitions: bool = False,
        ):
            """Initialize MessagesOrQueryResponseInputValidator."""
            super().__init__(
                error_target,
                requires_query,
                optional_tool_definitions,
                check_for_unsupported_tools,
            )
            self.enforce_tool_definitions = enforce_tool_definitions

        @override
        def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
            """Validate evaluation input, accepting either ``messages`` or ``query`` + ``response``."""
            has_messages = eval_input.get("messages") is not None
            has_query_or_response = (
                eval_input.get("query") is not None or eval_input.get("response") is not None
            )

            if has_messages and has_query_or_response:
                raise EvaluationException(
                    message="Cannot provide both 'messages' and 'query'/'response' inputs at the same time.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )

            if has_messages:
                messages = eval_input["messages"]
                if not isinstance(messages, list) or len(messages) == 0:
                    raise EvaluationException(
                        message="messages must be a non-empty list of message dictionaries.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )

                valid_roles = {role.value for role in MessageRole}
                roles_present: set = set()
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise EvaluationException(
                            message=f"Each item in 'messages' must be a dictionary at index {index}.",
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    role = message.get("role")
                    if role not in valid_roles:
                        raise EvaluationException(
                            message=(
                                f"Invalid role '{role}' at message index {index}. "
                                f"Valid roles are: {sorted(valid_roles)}."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    roles_present.add(role)

                if (
                    MessageRole.USER.value not in roles_present
                    or MessageRole.ASSISTANT.value not in roles_present
                ):
                    raise EvaluationException(
                        message="messages must contain at least one 'user' message and one 'assistant' message.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                return True

            return ConversationValidator.validate_eval_input(self, eval_input)


if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":  # pragma: no cover
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


logger = logging.getLogger(__name__)


def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes COHERENCE_AUDIO_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["COHERENCE_AUDIO_EVALUATOR"] = "CoherenceAudioEvaluator"
    return Enum("ExtendedErrorTarget", existing_members)


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class CoherenceAudioEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluate coherence of agent responses across a multi-turn audio conversation.

    The Coherence Audio evaluator assesses the logical flow, cross-turn context retention,
    and consistency of an AI agent's spoken responses across a full audio conversation by
    evaluating the whole-call audio recording alongside optional system/developer instructions
    and tool execution traces on a 1–5 scale.

    :param model_config: Configuration for the Azure OpenAI model or AI project.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration, ~azure.ai.evaluation.AzureAIProject, dict]
    :param threshold: The threshold for evaluation. Default is 3.
    :type threshold: Optional[int]
    """

    _PROMPTY_FILE = "coherence_multi_turn.prompty"
    _MULTI_TURN_PROMPTY_FILE = "coherence_multi_turn.prompty"
    _RESULT_KEY = "coherence"
    _OPTIONAL_PARAMS = ["messages", "audio_path"]
    _RESULT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "status", "reason", "properties"],
        "properties": {
            "score": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, None]},
            "status": {"type": "string", "enum": ["completed", "skipped"]},
            "reason": {"type": "string"},
            "properties": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "required": ["gating_summary", "conversation_flow_summary", "agent_coherence_issues"],
                "properties": {
                    "gating_summary": {"type": "string"},
                    "conversation_flow_summary": {"type": "string"},
                    "agent_coherence_issues": {"type": "string"},
                },
            },
        },
    }

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/coherence_audio"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold: int = 3, credential=None, evaluation_level=None, **kwargs):
        """Initialize the CoherenceAudioEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration, dict]
        :keyword threshold: The threshold for evaluation. Default is 3.
        :paramtype threshold: int
        :keyword credential: The credential for authenticating to Azure AI service.
        :paramtype credential: Optional[~azure.core.credentials.TokenCredential]
        :keyword evaluation_level: Evaluation level ("conversation" only for audio evaluation).
        :paramtype evaluation_level: Optional[Union[str, ~azure.ai.evaluation._common.constants.EvaluationLevel]]
        :keyword kwargs: Additional arguments to pass to the base evaluator.
        :paramtype kwargs: Any
        """
        judge_transport = kwargs.pop("judge_transport", "realtime_audio") or "realtime_audio"
        if judge_transport != "realtime_audio":
            raise ValueError(f"Unsupported Coherence Audio judge_transport: {judge_transport}")
        max_output_tokens = kwargs.pop("max_output_tokens", 3_000)
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer.")

        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)

        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR
        )
        if self._evaluation_level == EvaluationLevel.TURN:
            raise ValueError("realtime_audio supports conversation-level evaluation only.")

        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            enforce_tool_definitions=False,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold,
            _higher_is_better=True,
            **kwargs,
        )

        multi_turn_prompty_path = os.path.join(current_dir, self._MULTI_TURN_PROMPTY_FILE)
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(model_config),
            self._DEFAULT_OPEN_API_VERSION,
            f"azure-ai-evaluation (type=evaluator subtype={self.__class__.__name__})",
        )
        self._multi_turn_flow = AsyncPrompty.load(
            source=multi_turn_prompty_path,
            model=prompty_model_config,
            token_credential=credential,
            is_reasoning_model=self._is_reasoning_model,
        )

        self._judge_transport = judge_transport
        self._realtime_client = RealtimeCoherenceClient(
            model_config,
            credential,
            logger,
            max_output_tokens=max_output_tokens,
        )
        self._audio_system_prompt, self._audio_user_template = self._load_prompty(
            Path(prompty_path)
        )

    @overload
    def __call__(
        self,
        *,
        audio_path: str,
        messages: Optional[List[dict]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate coherence for a full multi-turn audio conversation.

        :keyword audio_path: Path to the whole-call WAV recording.
        :paramtype audio_path: str
        :keyword messages: Optional conversation messages including developer/system instructions,
            tool calls, tool results, and optional spoken turn placeholders. Can be empty or omitted
            for pure audio conversations without tool calls.
        :paramtype messages: Optional[List[dict]]
        :return: A dictionary containing coherence evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Invoke the instance using the overloaded __call__ signature.

        For detailed parameter types and return value documentation, see the overloaded __call__ definition.
        """
        return super().__call__(*args, **kwargs)

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation."""
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        return eval_input.get("messages") is not None or eval_input.get("audio_path") is not None

    def _build_result(
        self,
        score: Optional[float],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized result dictionary."""
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        metadata = {
            "prompt_tokens": p.get("input_token_count", 0),
            "completion_tokens": p.get("output_token_count", 0),
            "total_tokens": p.get("total_token_count", 0),
            "finish_reason": p.get("finish_reason", ""),
            "model": p.get("model_id", ""),
            "sample_input": p.get("sample_input", ""),
            "sample_output": p.get("sample_output", ""),
        }
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_passed": result == "pass" if result in ["pass", "fail"] else None,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_properties": {**properties, **metadata},
        }
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

    def _return_not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a result indicating that the conversation is not applicable for evaluation."""
        token_metadata = self._get_token_metadata({})
        result = {
            f"{self._result_key}": None,
            f"{self._result_key}_score": None,
            f"{self._result_key}_passed": None,
            f"{self._result_key}_result": "not_applicable",
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_status": "skipped",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_properties": None,
        }
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict:
        """Extract token usage and model metadata from the prompty output dict."""
        return {
            "prompt_tokens": prompty_output.get("input_token_count", 0),
            "completion_tokens": prompty_output.get("output_token_count", 0),
            "total_tokens": prompty_output.get("total_token_count", 0),
            "finish_reason": prompty_output.get("finish_reason", ""),
            "model": prompty_output.get("model_id", ""),
            "sample_input": prompty_output.get("sample_input", ""),
            "sample_output": prompty_output.get("sample_output", ""),
        }

    @override
    async def _real_call(self, **kwargs):
        """Convert inputs into a list of evaluable inputs, validate, and evaluate each turn."""
        messages = kwargs.get("messages")
        if messages is not None and not isinstance(messages, list):
            raise EvaluationException(
                message="messages must be a list of message dictionaries.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            )
        validation_input = {
            **kwargs,
            "messages": self._with_audio_roles_for_validation(messages or []),
        }
        self._validator.validate_eval_input(validation_input)

        return await self._the_super_real_call(**kwargs)

    @staticmethod
    def _with_audio_roles_for_validation(messages: List[dict]) -> List[dict]:
        """Satisfy text-role validation without changing the judge transcript."""
        roles = {message.get("role") for message in messages if isinstance(message, dict)}
        validation_messages = list(messages)
        if MessageRole.USER.value not in roles:
            validation_messages.append({"role": "user", "content": "[Spoken user turns are in audio.]"})
        if MessageRole.ASSISTANT.value not in roles:
            validation_messages.append(
                {"role": "assistant", "content": "[Spoken agent turns are in audio.]"}
            )
        return validation_messages

    async def _the_super_real_call(self, **kwargs):
        """Execute the core evaluation loop from EvaluatorBase._real_call."""
        eval_input_list = self._convert_kwargs_to_eval_input(**kwargs)
        per_turn_results = []
        for eval_input in eval_input_list:
            result = await self._do_eval(eval_input)
            if (
                result is not None
                and self._threshold is not None
                and result.get(f"{self._result_key}_status") != "skipped"
            ):
                score = result.get(f"{self._result_key}")
                result[f"{self._result_key}_threshold"] = self._threshold
                if self._higher_is_better:
                    if score >= self._threshold:
                        result[f"{self._result_key}_result"] = EVALUATION_PASS_FAIL_MAPPING[True]
                        result[f"{self._result_key}_passed"] = True
                    else:
                        result[f"{self._result_key}_result"] = EVALUATION_PASS_FAIL_MAPPING[False]
                        result[f"{self._result_key}_passed"] = False
                else:  # pragma: no cover
                    if score <= self._threshold:
                        result[f"{self._result_key}_result"] = EVALUATION_PASS_FAIL_MAPPING[True]
                        result[f"{self._result_key}_passed"] = True
                    else:
                        result[f"{self._result_key}_result"] = EVALUATION_PASS_FAIL_MAPPING[False]
                        result[f"{self._result_key}_passed"] = False
            per_turn_results.append(result)

        if len(per_turn_results) == 1:
            return per_turn_results[0]
        if len(per_turn_results) > 1:  # pragma: no cover
            return self._aggregate_results(per_turn_results=per_turn_results)
        return {}  # pragma: no cover

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a coherence audio evaluation."""
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)
        return await self._do_eval_conversation_level(eval_input)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Evaluate coherence at the conversation level using the Realtime audio path."""
        raw_messages = eval_input.get("messages") or []
        messages = _preprocess_messages(raw_messages) if raw_messages else []
        conversation_text = (
            serialize_messages(messages)
            or "[AUDIO: The complete call recording supplies the spoken user and agent turns.]"
        )

        audio_path = eval_input.get("audio_path")
        if not audio_path:
            raise EvaluationException(
                message="audio_path is required when judge_transport is realtime_audio.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            )

        user_prompt = self._render_user_prompt(self._audio_user_template, conversation_text)
        pcm = await asyncio.to_thread(self._read_pcm, Path(audio_path))
        response = await self._realtime_client.evaluate(
            pcm,
            self._audio_system_prompt,
            user_prompt,
            self._RESULT_SCHEMA,
        )
        usage = response.usage
        prompty_output_dict = {
            "llm_output": response.evaluation,
            "input_token_count": usage.get("input_tokens", 0),
            "output_token_count": usage.get("output_tokens", 0),
            "total_token_count": usage.get("total_tokens", 0),
            "finish_reason": response.status,
            "model_id": response.model,
            "sample_input": user_prompt,
            "sample_output": json.dumps(response.evaluation, ensure_ascii=False),
        }
        return self._parse_prompty_output(prompty_output_dict)

    @staticmethod
    def _load_prompty(path: Path) -> Tuple[str, str]:
        """Load the system and user sections from the multi-turn Prompty."""
        raw = path.read_text(encoding="utf-8")
        body = raw.split("---", 2)[2] if raw.startswith("---") else raw
        system, user = body.split("\nuser:\n", 1)
        return system.split("system:\n", 1)[1].strip(), user.strip()

    @staticmethod
    def _render_user_prompt(template: str, messages: str) -> str:
        """Render the messages input declared by the multi-turn Prompty."""
        return template.replace("{{messages}}", messages)

    @staticmethod
    def _read_pcm(path: Path) -> bytes:
        """Read a PCM16 WAV file, downmixing and resampling to mono 24-kHz PCM16 as needed."""
        with wave.open(str(path), "rb") as audio:
            channels, width, rate, frames, comptype, _ = audio.getparams()
            if channels < 1 or width != 2 or rate < 1 or comptype != "NONE":
                raise ValueError("Realtime Coherence requires an uncompressed PCM16 WAV.")
            pcm = audio.readframes(frames)
        if not pcm:
            raise ValueError("Realtime Coherence audio must not be empty.")
        if channels == 1 and rate == 24_000:
            return pcm
        samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).astype(np.float32).mean(axis=1)
        if rate != 24_000:
            factor = math.gcd(rate, 24_000)
            samples = resample_poly(samples, 24_000 // factor, rate // factor)
        return np.clip(np.rint(samples), -32768, 32767).astype("<i2").tobytes()

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[float, str]]:
        """Parse the prompty output into a standardized evaluation result."""
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            )

        llm_status = llm_output.get("status", "completed")
        if llm_status == "skipped":
            reason = llm_output.get("reason", "")
            return self._return_not_applicable_result(reason, self._threshold)

        score = llm_output.get("score", 1)
        if score is None:
            raise EvaluationException(
                message="Evaluator returned a null score.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.COHERENCE_AUDIO_EVALUATOR,
            )
        score = float(score)
        success_result = "pass" if score >= float(self._threshold) else "fail"
        reason = llm_output.get("reason", "")
        llm_properties = llm_output.get("properties", {}) or {}
        token_metadata = self._get_token_metadata(prompty_output_dict)
        llm_properties.update(token_metadata)
        result = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_passed": success_result == "pass",
            f"{self._result_key}_result": success_result,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": "completed",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": llm_properties,
        }
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result
