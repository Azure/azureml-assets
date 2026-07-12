# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from enum import Enum
from typing import Dict, Union, List, Optional, Any, Tuple

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    validate_model_config,
    reformat_conversation_history,
    reformat_agent_response,
    reformat_tool_definitions,
)
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ConversationValidator,
    ToolDefinitionsValidator,
)

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

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


if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TASK_COMPLETION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TASK_COMPLETION_EVALUATOR"] = "TaskCompletionEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class TaskCompletionEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Task Completion evaluator determines whether an AI agent successfully completed the requested task.

    This evaluator assesses task completion based on:
        - Final outcome and deliverable of the task
        - Completeness of task requirements

    This evaluator focuses solely on task completion and success, not on task adherence or intent understanding.

    Scoring is binary:
    - 1 (Pass): Task fully completed with usable deliverable that meets all user requirements
    - 0 (Fail): Task incomplete, partially completed, or deliverable does not meet requirements

    The evaluation includes task requirement analysis, outcome assessment, and completion gap identification.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a TaskCompletionEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call TaskCompletionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "task_completion.prompty"
    _MULTI_TURN_PROMPTY_FILE = "task_completion_multi_turn.prompty"
    _RESULT_KEY = "task_completion"
    _OPTIONAL_PARAMS = ["messages", "tool_definitions"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/task_completion"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, evaluation_level=None, **kwargs):
        """Initialize the TaskCompletionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword evaluation_level: Force a specific evaluation level for this invocation. When ``None``
            (default), the level is auto-detected from input shape (``messages`` -> conversation,
            ``query``/``response`` -> turn). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TURN`` to override auto-detection.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 1)

        # Validate and store evaluation level
        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR
        )

        # Initialize input validator (supports both query/response and messages)
        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
            enforce_tool_definitions=True,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold_value,
            **kwargs,
        )

        # Load the multi-turn prompty flow for multi-turn evaluation
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
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task completion for last agent response given a query, response, and optionally tool definitions.

        The query and response can be either a string or a list of messages.

        Example with string inputs and no tools:
            evaluator = TaskCompletionEvaluator(model_config)
            query = "Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine."
            response = "**Day 1:** Morning: Louvre Museum, Lunch: Le Comptoir du Relais..."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = TaskCompletionEvaluator(model_config)
            query = [{'role': 'system', 'content': 'You are a helpful travel planning assistant.'},
                     {'createdAt': 1700000060, 'role': 'user',
                     'content': [{'type': 'text', 'text': 'Plan a 3-day Paris itinerary with cultural
                     landmarks and cuisine'}]}]
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
                        'content': [{'type': 'text', 'text': '*Day 1:* Morning: Visit Louvre Museum (9 AM-12 PM)'}]}]
            tool_definitions = [{'name': 'get_attractions', 'description': 'Get tourist attractions for a city.',
                                'parameters': {'type': 'object', 'properties':
                                  {'city': {'type': 'string', 'description': 'City name.'}}}}]

            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task completion for a full multi-turn conversation.

        Example with messages and tool definitions:
            evaluator = TaskCompletionEvaluator(model_config)
            messages = [
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": [{"type": "text", "text": "Find flights to London next Friday"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search for flights to London."},
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "search_flights",
                            "arguments": {"origin": "NYC", "destination": "London", "date": "2025-01-10"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": {"flights": [{"airline": "BA", "price": "$450", "departure": "8:00 AM"}]},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I found a British Airways flight at $450 departing 8:00 AM. Shall I book it?",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Yes, book it."}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Done! Your flight is booked. Confirmation #BA12345."},
                    ],
                },
            ]
            tool_definitions = [
                {
                    "name": "search_flights",
                    "description": "Search for flights between two cities on a given date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "Departure city"},
                            "destination": {"type": "string", "description": "Arrival city"},
                            "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)"},
                        },
                    },
                }
            ]
            result = evaluator(messages=messages, tool_definitions=tool_definitions)

        :keyword messages: The full multi-turn conversation as a list of message dicts.
        :paramtype messages: List[dict]
        :keyword tool_definitions: An optional list of tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

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

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation.

        When ``_evaluation_level`` is set, it takes precedence. Otherwise, auto-detect
        based on whether ``messages`` is present in the input.

        :param eval_input: The evaluation input.
        :type eval_input: Dict
        :return: True if conversation-level evaluation should be used.
        :rtype: bool
        """
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        # Auto-detect (_evaluation_level is None)
        return eval_input.get("messages") is not None

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized result dictionary.

        :param score: The evaluation score (1, 0, or None).
        :param result: The result label ("pass", "fail", "skipped", or "error").
        :param reason: The reasoning or explanation string.
        :param status: The evaluation status ("completed", "skipped", or "error").
        :param properties: The properties dictionary.
        :param prompty_output_dict: Optional raw prompty output for extracting token metadata.
        :return: The standardized result dictionary.
        """
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
            f"{self._result_key}_properties": {**properties, **metadata}
        }
        # Add top-level token metadata fields for backward compatibility.
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

    def _return_not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict, None]]:
        """Return a result indicating that the tool call is not applicable for evaluation.

        :param error_message: The error message indicating why the evaluation is not applicable.
        :type error_message: str
        :param threshold: The threshold value for the evaluation.
        :type threshold: Union[int, float]
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float, None]]
        """
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
        # Add top-level token metadata fields for backward compatibility.
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
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Reshape inputs based on evaluation level before validation
        if self._evaluation_level == EvaluationLevel.CONVERSATION and not kwargs.get("messages"):
            query = kwargs.get("query")
            response = kwargs.get("response")
            if isinstance(query, str) and isinstance(response, str) and query and response:
                query, response = _wrap_string_messages(query, response)
            if isinstance(query, list) and isinstance(response, list):
                kwargs["messages"] = _merge_query_response_messages(query, response)
        elif self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            if any(m.get("role") == MessageRole.USER for m in kwargs["messages"]):
                query_messages, response_messages = _split_messages_at_latest_user(kwargs["messages"])
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)

        self._validator.validate_eval_input(kwargs)

        return await self._the_super_real_call(**kwargs)

    async def _the_super_real_call(self, **kwargs):
        """Perform the asynchronous call where real end-to-end evaluation logic runs.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Convert inputs into list of evaluable inputs.
        try:
            eval_input_list = self._convert_kwargs_to_eval_input(**kwargs)
        except Exception as e:
            logger.error(f"Error converting kwargs to eval_input_list: {e}")
            raise e
        per_turn_results = []
        # Evaluate all inputs.
        for eval_input in eval_input_list:
            result = await self._do_eval(eval_input)
            # logic to determine threshold pass/fail
            # if it wasn't computed in _do_eval
            try:
                keys = list(result.keys())
                contains_result_key = any(key.endswith("_result") for key in keys)
                contains_threshold_key = any(key.endswith("_threshold") for key in keys)
                if not contains_result_key or not contains_threshold_key:
                    for key in keys:
                        if key.endswith("_score"):
                            score_value = result[key]
                            base_key = key[:-6]  # Remove "_score" suffix
                            result_key = f"{base_key}_result"
                            threshold_key = f"{base_key}_threshold"
                            threshold_value = (
                                self._threshold.get(base_key) if isinstance(self._threshold, dict) else self._threshold
                            )
                            if not isinstance(threshold_value, (int, float)):
                                raise EvaluationException(
                                    "Threshold value must be a number.",
                                    internal_message=str(threshold_value),
                                    target=ErrorTarget.EVALUATE,
                                    category=ErrorCategory.INVALID_VALUE,
                                    blame=ErrorBlame.USER_ERROR,
                                )

                            if not contains_threshold_key:
                                result[threshold_key] = threshold_value

                            if not contains_result_key:
                                if self._higher_is_better:
                                    if float(score_value) >= threshold_value:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[True]
                                    else:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[False]
                                else:
                                    if float(score_value) <= threshold_value:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[True]
                                    else:
                                        result[result_key] = EVALUATION_PASS_FAIL_MAPPING[False]
            except Exception as e:
                logger.warning(f"Error calculating binary result: {e}")
            per_turn_results.append(result)
        # Return results as-is if only one result was produced.

        if len(per_turn_results) == 1:
            return per_turn_results[0]
        if len(per_turn_results) == 0:
            return {}  # TODO raise something?
        # Otherwise, aggregate results.
        return self._aggregate_results(per_turn_results=per_turn_results)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        """Do Task Completion evaluation.

        Routes to conversation-level or turn-level evaluation based on
        ``_evaluation_level`` (if set)
        or auto-detects from input shape (default).

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        # Single-turn path (query/response)
        if eval_input.get("query") is None or eval_input.get("response") is None:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Task Completion evaluator.",
                internal_message="Both query and response must be provided as input to the Task Completion evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
            )
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger, include_system_messages=True)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)
        if "tool_definitions" in eval_input and eval_input["tool_definitions"]:
            eval_input["tool_definitions"] = reformat_tool_definitions(eval_input["tool_definitions"], logger)

        # Remove keys not consumed by the single-turn prompty to avoid leaking extra kwargs
        eval_input.pop("messages", None)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate task completion for a full conversation-level evaluation.

        :param eval_input: The input containing ``messages`` and optionally ``tool_definitions``.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        messages = eval_input["messages"]

        messages = _preprocess_messages(messages)
        conversation_text = serialize_messages(messages)

        prompty_kwargs: Dict[str, Any] = {"messages": conversation_text}
        tool_definitions = eval_input.get("tool_definitions")
        if tool_definitions:
            prompty_kwargs["tool_definitions"] = reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._multi_turn_flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_kwargs)
        return self._parse_prompty_output(prompty_output_dict)

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[int, str]]:
        """Parse the prompty output into a standardized result dictionary.

        Shared between single-turn and multi-turn evaluation paths.
        Expects the canonical schema: score (int), reason (str), status (str), properties (dict|null).

        :param prompty_output_dict: Raw output from the prompty flow.
        :type prompty_output_dict: Dict
        :return: The parsed evaluation result.
        :rtype: Dict
        """
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
            )

        # Handle skipped status from LLM
        llm_status = llm_output.get("status", "completed")
        if llm_status == "skipped":
            reason = llm_output.get("reason", "")
            return self._return_not_applicable_result(reason, self._threshold)

        score = float(llm_output.get("score", 0))
        success_result = "pass" if score >= 1.0 else "fail"
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
        # Add top-level token metadata fields for backward compatibility.
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result
