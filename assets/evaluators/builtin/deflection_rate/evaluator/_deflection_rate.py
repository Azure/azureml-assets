# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from enum import Enum
from typing import Any, Dict, Optional, Union, List, Tuple

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    reformat_agent_response,
    validate_model_config,
)
from azure.ai.evaluation._common._experimental import experimental

from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
    ToolDefinitionsValidator,
)

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _is_intermediate_response, _preprocess_messages
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (
        _is_intermediate_response,
        _preprocess_messages,
    )

# Re-exported so the module keeps exposing the message-preprocessing helpers used
# by the test suite; they are invoked indirectly through _preprocess_messages.
try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._evaluators._common._validators import MessageRole
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    # azure-ai-evaluation 1.18.1 MessageRole; the 1.17.x SDK enum omits DEVELOPER,
    # which serialize_messages below relies on.
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
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    # Bodies below are copied from azure-ai-evaluation 1.18.1 (the earliest release
    # that ships these symbols). The only change is that serialize_messages uses the
    # module-level MessageRole above so the DEVELOPER role stays available on 1.17.x.
    from azure.ai.evaluation._common.utils import (
        _extract_text_from_content,
        _get_agent_response,
        _pretty_format_conversation_history,
    )

    class EvaluationLevel(str, Enum):
        """Supported evaluation levels for multi-turn evaluators.

        - ``CONVERSATION``: Force conversation-level evaluation using the multi-turn path.
        - ``TURN``: Force turn-level evaluation using the single-turn query/response path.
        """

        CONVERSATION = "conversation"
        TURN = "turn"

    def _merge_query_response_messages(query: List[dict], response: List[dict]) -> List[dict]:
        """Merge query and response message lists into a single conversation.

        :param query: The query messages.
        :type query: List[dict]
        :param response: The response messages.
        :type response: List[dict]
        :return: The merged conversation messages.
        :rtype: List[dict]
        """
        return [*query, *response]

    def _split_messages_at_latest_user(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
        """Split messages into query/response slices at the latest user turn.

        :param messages: The conversation messages.
        :type messages: List[dict]
        :return: A tuple of (query_messages, response_messages).
        :rtype: Tuple[List[dict], List[dict]]
        """
        latest_user_index = max(
            (i for i, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        if latest_user_index == -1:
            raise ValueError("messages must contain at least one message with role 'user'.")
        return messages[: latest_user_index + 1], messages[latest_user_index + 1:]

    def _wrap_string_messages(query: str, response: str) -> Tuple[List[dict], List[dict]]:
        """Wrap string query/response into separate message lists.

        :param query: The query string.
        :type query: str
        :param response: The response string.
        :type response: str
        :return: A tuple of (query_messages, response_messages).
        :rtype: Tuple[List[dict], List[dict]]
        """
        return (
            [{"role": "user", "content": [{"type": "text", "text": query}]}],
            [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
        )

    def _resolve_evaluation_level(
        evaluation_level: Optional[Union[EvaluationLevel, str]],
        error_target: ErrorTarget,
    ) -> Optional[EvaluationLevel]:
        """Validate and normalize the evaluation_level parameter.

        :param evaluation_level: The evaluation level to resolve.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :param error_target: The error target for exceptions.
        :type error_target: ErrorTarget
        :return: The resolved EvaluationLevel or None for auto-detect.
        :rtype: Optional[EvaluationLevel]
        """
        valid = [level.value for level in EvaluationLevel]
        if evaluation_level is None or evaluation_level == "":
            return None
        if isinstance(evaluation_level, EvaluationLevel):
            return evaluation_level
        if isinstance(evaluation_level, str):
            try:
                return EvaluationLevel(evaluation_level)
            except ValueError as exc:
                raise EvaluationException(
                    message=(f"Invalid evaluation_level '{evaluation_level}'. " f"Must be one of: {valid}."),
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=error_target,
                ) from exc
        raise EvaluationException(
            message=(f"Invalid evaluation_level '{evaluation_level}'. " f"Must be one of: {valid}."),
            blame=ErrorBlame.USER_ERROR,
            category=ErrorCategory.INVALID_VALUE,
            target=error_target,
        )

    def serialize_messages(messages):
        """Serialize a list of chat messages into a labeled text transcript for multi-turn prompts.

        **Input format:** List of message dicts, each with ``"role"`` (``user``, ``assistant``, ``tool``,
        ``system``, ``developer``) and ``"content"`` (string or list of content-block dicts like
        ``{"type": "text", "text": "..."}``). Tool messages may include ``tool_call_id`` and content
        blocks of type ``tool_result``/``tool_call``.

        **Output format:** Plain-text transcript with labeled turns::

            User turn 1:
              <user text>

            Agent turn 1:
              <assistant text>
              [TOOL_CALL] func_name({"arg": "val"})
              [TOOL_RESULT] <result>

            User turn 2:
              <user text>
            ...

        System/developer messages are included as a system preamble. Consecutive messages of the same
        role are grouped into a single turn. Assistant string content is auto-normalized to content-block
        format for consistent formatting.

        :param messages: Chat messages with role and content.
        :type messages: List[dict]
        :return: Formatted text transcript.
        :rtype: str
        """
        if not messages:
            return ""

        # Uses the module-level MessageRole above (the 1.17.x SDK enum omits DEVELOPER).
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

            # _get_agent_response expects content as list of dicts, not a plain string
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
                if isinstance(content, str):
                    text_in_msg = [content]
                else:
                    text_in_msg = _extract_text_from_content(content)
                if text_in_msg:
                    cur_user_query.append(text_in_msg)

            elif role in (MessageRole.ASSISTANT, MessageRole.TOOL):
                if cur_user_query:
                    all_user_queries.append(cur_user_query)
                    cur_user_query = []
                cur_agent_response.append(normalized)

        # Flush any remaining buffered turn
        if cur_user_query:
            all_user_queries.append(cur_user_query)
        if cur_agent_response:
            formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
            all_agent_responses.append([formatted])

        conversation_history: Dict = {
            "user_queries": all_user_queries,
            "agent_responses": all_agent_responses[: len(all_user_queries) - 1] if len(all_user_queries) > 0 else [],
        }
        if system_message:
            conversation_history["system_message"] = system_message

        result = _pretty_format_conversation_history(conversation_history)

        # Append any trailing agent turn (the final response after the last user query)
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
        """Validator that supports both single-turn (query/response) and multi-turn (messages) inputs.

        A single implementation serves all evaluators via a behavior flag:
          - ``enforce_tool_definitions`` (default False): validate ``tool_definitions`` in both the
            messages path and the query/response path. Set True for evaluators that require
            tool definitions.
        """

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
            super().__init__(error_target, requires_query, optional_tool_definitions, check_for_unsupported_tools)
            self.enforce_tool_definitions = enforce_tool_definitions

        @override
        def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
            """Validate evaluation input, supporting messages as an alternative to query/response."""
            # Multi-turn path (messages list)
            messages = eval_input.get("messages")
            if messages is not None:
                if not isinstance(messages, list):
                    raise EvaluationException(
                        message="messages must be provided as a list of message dictionaries.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if len(messages) == 0:
                    raise EvaluationException(
                        message="messages list must not be empty.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )

                # Per-message structural checks
                valid_roles = {role.value for role in MessageRole}
                roles_present: set = set()
                for index, message in enumerate(messages):
                    if not isinstance(message, dict):
                        raise EvaluationException(
                            message=(
                                "Each item in 'messages' must be a dictionary, "
                                f"but item at index {index} is {type(message).__name__}."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    role = message.get("role")
                    if role is None:
                        raise EvaluationException(
                            message=(
                                "Each message must contain a 'role' key, "
                                f"but message at index {index} is missing it."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    if role not in valid_roles:
                        raise EvaluationException(
                            message=(
                                f"Invalid role '{role}' at message index {index}. "
                                f"Must be one of: {sorted(valid_roles)}."
                            ),
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=self.error_target,
                        )
                    roles_present.add(role)

                # Conversation-level checks
                if MessageRole.USER.value not in roles_present:
                    raise EvaluationException(
                        message="messages must contain at least one message with role 'user'.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if MessageRole.ASSISTANT.value not in roles_present:
                    raise EvaluationException(
                        message="messages must contain at least one message with role 'assistant'.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )

                if self.enforce_tool_definitions:
                    tool_definitions = eval_input.get("tool_definitions")
                    tool_definitions_validation_exception = self._validate_tool_definitions(tool_definitions)
                    if tool_definitions_validation_exception:
                        raise tool_definitions_validation_exception
                return True

            if self.enforce_tool_definitions:
                return super().validate_eval_input(eval_input)
            return ConversationValidator.validate_eval_input(self, eval_input)


logger = logging.getLogger(__name__)


# Use the SDK's ErrorTarget member when the installed version defines it; otherwise fall back to EVALUATE.
_ERROR_TARGET = getattr(ErrorTarget, "DEFLECTION_RATE_EVALUATOR", ErrorTarget.EVALUATE)


@experimental
class DeflectionRateEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Deflection Rate evaluator determines whether an AI assistant deflected a user query.

    A deflection occurs when the AI indicates the topic is out of scope, suggests seeking help
    elsewhere, or fails to provide a direct answer. This evaluator is useful for measuring:
        - Chatbot effectiveness in resolving queries without human intervention
        - Customer support automation rates
        - Self-service success rates

    Deflection types:
        - plain_denial: Explicitly states it cannot answer
        - send_elsewhere: Suggests seeking help from another source
        - reframe: Reframes the question to fall within its scope
        - plain_answer: Provides a direct answer (no deflection)

    Scoring is binary:
    - 0: No deflection - the system provided a direct answer
    - 1: Deflection - the system indicated the topic is out of scope

    Note: Lower scores are better for this evaluator (desirable_direction: decrease).

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START deflection_rate_evaluator]
            :end-before: [END deflection_rate_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a DeflectionRateEvaluator with a response.

    """

    _PROMPTY_FILE = "deflection_rate.prompty"
    _MULTI_TURN_PROMPTY_FILE = "deflection_rate_multi_turn.prompty"
    _RESULT_KEY = "deflection_rate"
    _OPTIONAL_PARAMS = ["messages"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/deflection_rate"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=0, evaluation_level=None, **kwargs):
        """Initialize the DeflectionRateEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 0 (no deflection expected).
        :type threshold: int
        :keyword evaluation_level: Force turn or conversation evaluation. When omitted, ``messages``
            selects conversation evaluation and ``response`` selects turn evaluation.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = False  # Lower deflection is better
        self._evaluation_level = _resolve_evaluation_level(evaluation_level, _ERROR_TARGET)

        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=_ERROR_TARGET,
            requires_query=False,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold,
            _higher_is_better=self._higher_is_better,
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

    @overload
    def __call__(
        self,
        *,
        response: Union[str, List[dict]],
    ) -> Dict[str, Union[str, int]]:
        """Evaluate deflection rate for a given response.

        The response can be either a string or a list of messages.

        Example with string input:
            evaluator = DeflectionRateEvaluator(model_config)
            response = "The capital of France is Paris."

            result = evaluator(response=response)

        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :return: A dictionary with the deflection rate evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
    ) -> Dict[str, Union[str, int]]:
        """Evaluate deflection rate across a complete conversation."""

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Invoke the instance using the overloaded __call__ signature.

        For detailed parameter types and return value documentation, see the overloaded __call__ definition.
        """
        return super().__call__(*args, **kwargs)

    def _return_not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a standardized result for an evaluation that was skipped."""
        return self._build_result(
            score=None,
            result="not_applicable",
            reason=f"Not applicable: {error_message}",
            status="skipped",
            properties={},
        )

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation."""
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        return eval_input.get("messages") is not None

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        if self._evaluation_level == EvaluationLevel.CONVERSATION and kwargs.get("messages") is None:
            raise EvaluationException(
                message="Messages must be provided for conversation-level Deflection Rate evaluation.",
                internal_message="Messages must be provided for conversation-level Deflection Rate evaluation.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=_ERROR_TARGET,
            )

        if self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            messages = kwargs["messages"]
            if isinstance(messages, list) and any(
                isinstance(message, dict) and message.get("role") == "user" for message in messages
            ):
                query_messages, response_messages = _split_messages_at_latest_user(messages)
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)

        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        """Do Deflection Rate evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if eval_input.get("response") is None:
            raise EvaluationException(
                message="Response must be provided as input to the Deflection Rate evaluator.",
                internal_message="Response must be provided as input to the Deflection Rate evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=_ERROR_TARGET,
            )

        # Check for intermediate response (function_call or mcp_approval_request)
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )

        # Reformat response if it's a list of messages
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
            eval_input["response"] = reformat_agent_response(
                eval_input["response"], logger, include_tool_messages=True
            )

        # The single-turn prompt evaluates only the response; query/messages are routing inputs.
        eval_input.pop("query", None)
        eval_input.pop("messages", None)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate deflection rate across a full serialized conversation."""
        messages = _preprocess_messages(eval_input["messages"])
        conversation_text = serialize_messages(messages)
        prompty_output_dict = await self._multi_turn_flow(
            timeout=self._LLM_CALL_TIMEOUT,
            messages=conversation_text,
        )
        return self._parse_prompty_output(prompty_output_dict)

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
        deflection_type: str = "",
    ) -> Dict[str, Any]:
        """Build a standardized result dictionary with legacy metadata fields."""
        metadata = self._get_token_metadata(prompty_output_dict or {})
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_passed": result == "pass" if result in ["pass", "fail"] else None,
            f"{self._result_key}_result": result,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": {**properties, **metadata},
            f"{self._result_key}_deflection_type": deflection_type,
        }
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

    @staticmethod
    def _get_token_metadata(prompty_output: Dict) -> Dict[str, Any]:
        """Extract token usage and model metadata from prompty output."""
        return {
            "prompt_tokens": prompty_output.get("input_token_count", 0),
            "completion_tokens": prompty_output.get("output_token_count", 0),
            "total_tokens": prompty_output.get("total_token_count", 0),
            "finish_reason": prompty_output.get("finish_reason", ""),
            "model": prompty_output.get("model_id", ""),
            "sample_input": prompty_output.get("sample_input", ""),
            "sample_output": prompty_output.get("sample_output", ""),
        }

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Any]:
        """Parse either prompt flow's output into the standardized result schema."""
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=_ERROR_TARGET,
            )

        llm_status = str(llm_output.get("status") or "completed").strip().lower()
        if llm_status == "skipped":
            reason = llm_output.get("reason", llm_output.get("explanation", ""))
            return self._return_not_applicable_result(reason, self._threshold)

        score_value = llm_output.get("score", 0)
        if isinstance(score_value, str):
            score = int(score_value) if score_value.isdigit() else 0
        else:
            score = int(score_value) if score_value else 0

        success_result = "pass" if score <= self._threshold else "fail"
        return self._build_result(
            score=score,
            result=success_result,
            reason=llm_output.get("explanation", llm_output.get("reason", "")),
            status="completed",
            properties=llm_output.get("properties") or {},
            prompty_output_dict=prompty_output_dict,
            deflection_type=llm_output.get("deflection_type", ""),
        )
