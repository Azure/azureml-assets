# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import math
from typing import Dict, List, Optional, Union, Any, Tuple

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._common.utils import (
    ErrorBlame,
    ErrorTarget,
    EvaluationException,
    ErrorCategory,
    construct_prompty_model_config,
    validate_model_config,
    _extract_text_from_content,
    _get_agent_response,
    _pretty_format_conversation_history,
)
from azure.ai.evaluation._common.utils import reformat_tool_definitions

from abc import ABC, abstractmethod
from enum import Enum


# region Validators


class ValidatorInterface(ABC):
    """Abstract base class defining the interface that all validators must implement."""

    @abstractmethod
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate the evaluation input dictionary."""
        pass


class MessageRole(str, Enum):
    """Valid message roles in conversations."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ContentType(str, Enum):
    """Valid content types in messages."""

    TEXT = "text"
    INPUT_TEXT = "input_text"
    OUTPUT_TEXT = "output_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FUNCTION_CALL = "function_call"
    FUNCTION_CALL_OUTPUT = "function_call_output"
    MCP_APPROVAL_REQUEST = "mcp_approval_request"
    MCP_APPROVAL_RESPONSE = "mcp_approval_response"
    OPENAPI_CALL = "openapi_call"
    OPENAPI_CALL_OUTPUT = "openapi_call_output"


class EvaluationLevel(str, Enum):
    """Supported evaluation levels for GroundednessEvaluator.

    - ``CONVERSATION``: Force conversation-level evaluation using the multi-turn path.
    - ``TRACE``: Force trace-level evaluation using the single-turn query/response path.
    """

    CONVERSATION = "conversation"
    TRACE = "trace"


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
    if evaluation_level is None:
        return None
    if isinstance(evaluation_level, EvaluationLevel):
        return evaluation_level
    if isinstance(evaluation_level, str):
        try:
            return EvaluationLevel(evaluation_level)
        except ValueError:
            raise EvaluationException(
                message=(
                    f"Invalid evaluation_level '{evaluation_level}'. "
                    f"Must be one of: {valid}."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=error_target,
            )
    raise EvaluationException(
        message=(
            f"Invalid evaluation_level '{evaluation_level}'. "
            f"Must be one of: {valid}."
        ),
        blame=ErrorBlame.USER_ERROR,
        category=ErrorCategory.INVALID_VALUE,
        target=error_target,
    )


def _merge_query_response_messages(query: List[dict], response: List[dict]) -> List[dict]:
    """Merge query and response message lists into a single conversation."""
    return [*query, *response]


def _split_messages_at_latest_user(messages: List[dict]) -> Tuple[List[dict], List[dict]]:
    """Split messages into query/response slices at the latest user turn."""
    latest_user_index = max(i for i, message in enumerate(messages) if message["role"] == MessageRole.USER)
    return messages[: latest_user_index + 1], messages[latest_user_index + 1:]


def _wrap_string_messages(query: str, response: str) -> Tuple[List[dict], List[dict]]:
    """Wrap string query/response into separate message lists."""
    return (
        [{"role": "user", "content": [{"type": "text", "text": query}]}],
        [{"role": "assistant", "content": [{"type": "text", "text": response}]}],
    )


class ConversationValidator(ValidatorInterface):
    """Validate conversation inputs (queries and responses) comprised of message lists."""

    requires_query: bool = True
    check_for_unsupported_tools: bool = False
    error_target: ErrorTarget

    UNSUPPORTED_TOOLS: List[str] = [
        "azure_ai_search",
        "bing_custom_search",
        "bing_grounding",
        "browser_automation",
        "code_interpreter_call",
        "computer_call",
        "azure_fabric",
        "openapi_call",
        "sharepoint_grounding",
        "web_search"
    ]

    def __init__(
        self,
        error_target: ErrorTarget,
        requires_query: bool = True,
        check_for_unsupported_tools: bool = False
    ):
        """Initialize with error target and query requirement."""
        self.requires_query = requires_query
        self.check_for_unsupported_tools = check_for_unsupported_tools
        self.error_target = error_target

    def _validate_field_exists(
        self, item: Dict[str, Any], field_name: str, context: str
    ) -> Optional[EvaluationException]:
        """Validate that a field exists in a dictionary."""
        if field_name not in item:
            return EvaluationException(
                message=f"Each {context} must contain a '{field_name}' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        return None

    def _validate_string_field(
        self, item: Dict[str, Any], field_name: str, context: str
    ) -> Optional[EvaluationException]:
        """Validate that a field exists and is a string."""
        if field_name not in item:
            return EvaluationException(
                message=f"Each {context} must contain a '{field_name}' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if not isinstance(item[field_name], str):
            return EvaluationException(
                message=f"The '{field_name}' field must be a string in {context}.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        return None

    def _validate_list_field(
        self, item: Dict[str, Any], field_name: str, context: str
    ) -> Optional[EvaluationException]:
        """Validate that a field exists and is a list."""
        if field_name not in item:
            return EvaluationException(
                message=f"Each {context} must contain a '{field_name}' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if not isinstance(item[field_name], list):
            return EvaluationException(
                message=f"The '{field_name}' field must be a list in {context}.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        return None

    def _validate_dict_field(
        self, item: Dict[str, Any], field_name: str, context: str
    ) -> Optional[EvaluationException]:
        """Validate that a field exists and is a dictionary."""
        if field_name not in item:
            return EvaluationException(
                message=f"Each {context} must contain a '{field_name}' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if not isinstance(item[field_name], dict):
            return EvaluationException(
                message=f"The '{field_name}' field must be a dictionary in {context}.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        return None

    def _validate_text_content_item(self, content_item: Dict[str, Any], role: str) -> Optional[EvaluationException]:
        """Validate a text content item."""
        if "text" not in content_item:
            return EvaluationException(
                message=f"Each content item must contain a 'text' field for message with role '{role}'.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if not isinstance(content_item["text"], str):
            return EvaluationException(
                message="The 'text' field must be a string in content items.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        return None

    def _validate_tool_call_content_item(self, content_item: Dict[str, Any]) -> Optional[EvaluationException]:
        """Validate a tool_call content item."""
        valid_tool_call_content_types = [
            ContentType.TOOL_CALL,
            ContentType.FUNCTION_CALL,
            ContentType.OPENAPI_CALL,
            ContentType.MCP_APPROVAL_REQUEST
        ]
        valid_tool_call_content_types_as_strings = [t.value for t in valid_tool_call_content_types]
        if "type" not in content_item or content_item["type"] not in valid_tool_call_content_types:
            return EvaluationException(
                message=(
                    f"The content item must be of type {valid_tool_call_content_types_as_strings} "
                    "in tool_call content item."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )

        if content_item["type"] == ContentType.MCP_APPROVAL_REQUEST:
            return None

        error = self._validate_string_field(content_item, "name", "tool_call content items")
        if error:
            return error
        error = self._validate_dict_field(content_item, "arguments", "tool_call content items")
        if error:
            return error
        error = self._validate_string_field(content_item, "tool_call_id", "tool_call content items")
        if error:
            return error
        return None

    def _validate_user_or_system_message(self, message: Dict[str, Any], role: str) -> Optional[EvaluationException]:
        """Validate user or system message content."""
        content = message["content"]
        if isinstance(content, list):
            for content_item in content:
                content_type = content_item["type"]
                if content_type not in [ContentType.TEXT, ContentType.INPUT_TEXT]:
                    return EvaluationException(
                        message=(
                            f"Invalid content type '{content_type}' for message with role '{role}'. "
                            f"Must be '{ContentType.TEXT.value}' or '{ContentType.INPUT_TEXT.value}'."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                error = self._validate_text_content_item(content_item, role)
                if error:
                    return error
        return None

    def _validate_assistant_message(self, message: Dict[str, Any]) -> Optional[EvaluationException]:
        """Validate assistant message content."""
        content = message["content"]
        if isinstance(content, list):
            valid_assistant_content_types = [
                ContentType.TEXT,
                ContentType.OUTPUT_TEXT,
                ContentType.TOOL_CALL,
                ContentType.FUNCTION_CALL,
                ContentType.MCP_APPROVAL_REQUEST,
                ContentType.OPENAPI_CALL
            ]
            valid_assistant_content_type_values = [t.value for t in valid_assistant_content_types]
            for content_item in content:
                content_type = content_item["type"]
                if content_type not in valid_assistant_content_types:
                    return EvaluationException(
                        message=(
                            f"Invalid content type '{content_type}' for message with "
                            f"role '{MessageRole.ASSISTANT.value}'. "
                            f"Must be one of {valid_assistant_content_type_values}."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if content_type in [ContentType.TEXT, ContentType.OUTPUT_TEXT]:
                    error = self._validate_text_content_item(content_item, MessageRole.ASSISTANT)
                    if error:
                        return error
                elif content_type in [ContentType.TOOL_CALL, ContentType.FUNCTION_CALL, ContentType.OPENAPI_CALL]:
                    error = self._validate_tool_call_content_item(content_item)
                    if error:
                        return error

                    # Raise error in case of unsupported tools for evaluators that enabled check_for_unsupported_tools
                    if self.check_for_unsupported_tools:
                        if content_type == ContentType.TOOL_CALL or content_type == ContentType.OPENAPI_CALL:
                            name = (
                                "openapi_call" if content_type == ContentType.OPENAPI_CALL
                                else content_item["name"].lower()
                            )
                            if name in self.UNSUPPORTED_TOOLS:
                                return EvaluationException(
                                    message=(
                                        f"{name} tool call is currently not supported for "
                                        f"{self.error_target.value} evaluator."
                                    ),
                                    blame=ErrorBlame.USER_ERROR,
                                    category=ErrorCategory.NOT_APPLICABLE,
                                    target=self.error_target,
                                )
        return None

    def _validate_tool_message(self, message: Dict[str, Any]) -> Optional[EvaluationException]:
        """Validate tool message content."""
        content = message["content"]
        if not isinstance(content, list):
            return EvaluationException(
                message=(
                    "The 'content' field must be a list of dictionaries messages "
                    f"for role '{MessageRole.TOOL.value}'."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        error = self._validate_string_field(
            message, "tool_call_id", f"content items for role '{MessageRole.TOOL.value}'"
        )
        if error:
            return error
        for content_item in content:
            content_type = content_item["type"]
            valid_tool_content_types = [
                ContentType.TOOL_RESULT,
                ContentType.FUNCTION_CALL_OUTPUT,
                ContentType.MCP_APPROVAL_RESPONSE,
                ContentType.OPENAPI_CALL_OUTPUT
            ]
            valid_tool_content_types_as_strings = [t.value for t in valid_tool_content_types]
            if content_type not in valid_tool_content_types:
                return EvaluationException(
                    message=(
                        f"Invalid content type '{content_type}' for message with role "
                        f"'{MessageRole.TOOL.value}'. Must be one of {valid_tool_content_types_as_strings}."
                    ),
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )

            if content_type in [
                ContentType.TOOL_RESULT, ContentType.OPENAPI_CALL_OUTPUT, ContentType.FUNCTION_CALL_OUTPUT
            ]:
                error = self._validate_field_exists(
                    content_item, content_type, f"content items for role '{MessageRole.TOOL.value}'"
                )
                if error:
                    return error
        return None

    def _validate_message_dict(self, message: Dict[str, Any]) -> Optional[EvaluationException]:
        """Validate a single message dictionary."""
        if "role" not in message:
            return EvaluationException(
                message="Each message must contain a 'role' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if "content" not in message:
            return EvaluationException(
                message="Each message must contain a 'content' field.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        role = message["role"]
        content = message["content"]
        content_is_string_or_list_of_dicts = isinstance(content, str) or (
            isinstance(content, list) and all(item and isinstance(item, dict) for item in content)
        )
        if not content_is_string_or_list_of_dicts:
            return EvaluationException(
                message="The 'content' field must be a string or a list of dictionaries messages.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if len(content) == 0:
            return EvaluationException(
                message="The 'content' field can't be empty.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if isinstance(content, list):
            all_messages_have_type_field = all("type" in item for item in content)
            if not all_messages_have_type_field:
                return EvaluationException(
                    message="Each content item in the 'content' list must contain a 'type' field.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
        if role in [MessageRole.USER, MessageRole.SYSTEM]:
            error = self._validate_user_or_system_message(message, role)
            if error:
                return error
        elif role == MessageRole.ASSISTANT:
            error = self._validate_assistant_message(message)
            if error:
                return error
        elif role == MessageRole.TOOL:
            error = self._validate_tool_message(message)
            if error:
                return error
        return None

    def _validate_input_messages_list(self, input_messages: Any, input_name: str) -> Optional[EvaluationException]:
        if input_messages is None:
            return EvaluationException(
                message=f"{input_name} is a required input and cannot be None.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        if isinstance(input_messages, str):
            if input_messages == "":
                return EvaluationException(
                    message=f"{input_name} string cannot be empty.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.MISSING_FIELD,
                    target=self.error_target,
                )
            return None
        if not isinstance(input_messages, list):
            return EvaluationException(
                message=f"{input_name} must be a string or a list of messages.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        if len(input_messages) == 0:
            return EvaluationException(
                message=f"{input_name} list cannot be empty.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        if not all(isinstance(message, dict) for message in input_messages):
            return EvaluationException(
                message=f"Each message in the {input_name.lower()} list must be a dictionary.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        for message in input_messages:
            error = self._validate_message_dict(message)
            if error:
                return error
        return None

    def _validate_conversation(self, conversation: Any) -> Optional[EvaluationException]:
        """Validate the conversation input."""
        if not isinstance(conversation, dict):
            return EvaluationException(
                message="Conversation must be a dictionary.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        error = self._validate_list_field(conversation, "messages", "Conversation")
        if error:
            return error
        messages = conversation["messages"]
        return self._validate_input_messages_list(messages, "Conversation messages")

    def _validate_query(self, query: Any) -> Optional[EvaluationException]:
        """Validate the query input."""
        if not self.requires_query:
            return None
        return self._validate_input_messages_list(query, "Query")

    def _validate_response(self, response: Any) -> Optional[EvaluationException]:
        """Validate the response input."""
        return self._validate_input_messages_list(response, "Response")

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate the evaluation input dictionary."""
        conversation = eval_input.get("conversation")
        if conversation:
            conversation_validation_exception = self._validate_conversation(conversation)
            if conversation_validation_exception:
                raise conversation_validation_exception
            return True
        query = eval_input.get("query")
        response = eval_input.get("response")
        query_validation_exception = self._validate_query(query)
        if query_validation_exception:
            raise query_validation_exception
        response_validation_exception = self._validate_response(response)
        if response_validation_exception:
            raise response_validation_exception
        return True


class MessagesOrQueryResponseInputValidator(ConversationValidator):
    """Validator that supports both single-turn (query/response) and multi-turn (messages) inputs.

    When ``messages`` is provided, it validates the messages list.
    Otherwise, it delegates to the parent ``ConversationValidator`` for the query/response path.
    """

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate evaluation input, supporting messages as an alternative to query/response."""
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
            valid_roles = {r.value for r in MessageRole}
            roles_present: set = set()
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    raise EvaluationException(
                        message=(
                            f"Each item in 'messages' must be a dictionary, "
                            f"but item at index {i} is {type(msg).__name__}."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                role = msg.get("role")
                if role is None:
                    raise EvaluationException(
                        message=f"Each message must contain a 'role' key, but message at index {i} is missing it.",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                if role not in valid_roles:
                    raise EvaluationException(
                        message=(
                            f"Invalid role '{role}' at message index {i}. "
                            f"Must be one of: {sorted(valid_roles)}."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                roles_present.add(role)

            # Conversation-level checks
            if MessageRole.USER not in roles_present:
                raise EvaluationException(
                    message="messages must contain at least one message with role 'user'.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            if MessageRole.ASSISTANT not in roles_present:
                raise EvaluationException(
                    message="messages must contain at least one message with role 'assistant'.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            if messages[-1]["role"] != MessageRole.ASSISTANT:
                raise EvaluationException(
                    message=(
                        f"The last message must have role 'assistant', "
                        f"but found role '{messages[-1]['role']}'."
                    ),
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            # The final assistant message must contain text
            last_content = messages[-1].get("content", "")
            if isinstance(last_content, list):
                has_text = any(
                    isinstance(c, dict) and c.get("type") in ("text",)
                    or isinstance(c, str)
                    for c in last_content
                )
                if not has_text:
                    raise EvaluationException(
                        message=(
                            "The last assistant message must contain text content, "
                            "not only tool calls. The conversation appears to be "
                            "mid-execution — provide the agent's final text response."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
            return True
        return super().validate_eval_input(eval_input)


# endregion Validators

try:
    from azure.ai.evaluation._user_agent import UserAgentSingleton
except ImportError:

    class UserAgentSingleton:
        """Fallback singleton for user agent when import fails."""

        @property
        def value(self) -> str:
            """Return the user agent value."""
            return "None"


logger = logging.getLogger(__name__)


# ```utils.py
def simplify_messages(messages, drop_system=True, drop_tool_calls=False, logger=None):
    """
    Simplify a list of conversation messages by keeping only role and content.

    Optionally filter out system messages and/or tool calls.

    :param messages: List of message dicts (e.g., from query or response)
    :param drop_system: If True, remove system role messages
    :param drop_tool_calls: If True, remove tool_call items from assistant content
    :return: New simplified list of messages
    """
    if isinstance(messages, str):
        return messages
    try:
        # Validate input is a list
        if not isinstance(messages, list):
            return messages

        simplified_msgs = []
        for msg in messages:
            # Ensure msg is a dict
            if not isinstance(msg, dict):
                simplified_msgs.append(msg)
                continue

            role = msg.get("role")
            content = msg.get("content", [])

            # Drop system message (if should)
            if drop_system and role == "system":
                continue

            # Simplify user messages
            if role == "user":
                simplified_msg = {
                    "role": role,
                    "content": _extract_text_from_content(content),
                }
                simplified_msgs.append(simplified_msg)
                continue

            # Drop tool results (if should)
            if drop_tool_calls and role == "tool":
                continue

            # Simplify assistant messages
            if role == "assistant":
                simplified_content = _extract_text_from_content(content)
                # Check if message has content
                if simplified_content:
                    simplified_msg = {"role": role, "content": simplified_content}
                    simplified_msgs.append(simplified_msg)
                    continue

                # Drop tool calls (if should)
                if drop_tool_calls and any(c.get("type") == "tool_call" for c in content if isinstance(c, dict)):
                    continue

            # If we reach here, it means we want to keep the message
            simplified_msgs.append(msg)

        return simplified_msgs

    except Exception as ex:
        if logger:
            logger.debug(f"Error simplifying messages: {str(ex)}. Returning original messages.")
        return messages


# ```


def _is_intermediate_response(response):
    """Check if response is intermediate (last content item is function_call or mcp_approval_request)."""
    if isinstance(response, list) and len(response) > 0:
        last_msg = response[-1]
        if isinstance(last_msg, dict) and last_msg.get("role") == "assistant":
            content = last_msg.get("content", [])
            if isinstance(content, list) and len(content) > 0:
                last_content = content[-1]
                if (isinstance(last_content, dict) and
                        last_content.get("type") in ("function_call", "mcp_approval_request")):
                    return True
    return False


def _drop_mcp_approval_messages(messages):
    """Remove MCP approval request/response messages."""
    if not isinstance(messages, list):
        return messages
    return [
        msg for msg in messages
        if not (
            isinstance(msg, dict)
            and isinstance(msg.get("content"), list)
            and (
                (msg.get("role") == "assistant" and any(
                    isinstance(c, dict) and c.get("type") == "mcp_approval_request" for c in msg["content"]))
                or (msg.get("role") == "tool" and any(
                    isinstance(c, dict) and c.get("type") == "mcp_approval_response" for c in msg["content"]))
            )
        )
    ]


def _normalize_function_call_types(messages):
    """Normalize function_call/function_call_output/openapi_call/openapi_call_output types to tool_call/tool_result."""
    if not isinstance(messages, list):
        return messages
    for msg in messages:
        if not isinstance(msg, dict) or not isinstance(msg.get("content"), list):
            continue
        for item in msg["content"]:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "function_call":
                item["type"] = "tool_call"
            elif t == "function_call_output":
                item["type"] = "tool_result"
                if "function_call_output" in item:
                    item["tool_result"] = item.pop("function_call_output")
            elif t == "openapi_call":
                item["type"] = "tool_call"
            elif t == "openapi_call_output":
                item["type"] = "tool_result"
                if "openapi_call_output" in item:
                    item["tool_result"] = item.pop("openapi_call_output")
    return messages


def _preprocess_messages(messages):
    """Drop MCP approval messages and normalize function call types."""
    messages = _drop_mcp_approval_messages(messages)
    messages = _normalize_function_call_types(messages)
    return messages


def serialize_messages(messages: List[dict]) -> str:
    """Serialize a list of chat messages into a labeled text transcript for the multi-turn prompty.

    **Input format:** List of message dicts, each with ``"role"`` (``user``, ``assistant``, ``tool``,
    ``system``) and ``"content"`` (string or list of content-block dicts like
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

    System messages are included as a system preamble. Consecutive messages of the same
    role are grouped into a single turn. Assistant string content is auto-normalized to content-block
    format for consistent formatting.

    :param messages: Chat messages with role and content.
    :type messages: List[dict]
    :return: Formatted text transcript.
    :rtype: str
    """
    if not messages:
        return ""

    all_user_queries: List = []
    all_agent_responses: List = []
    cur_user_query: List = []
    cur_agent_response: List = []
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

        if role == MessageRole.SYSTEM:
            system_message = msg.get("content", "")

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

    if cur_user_query:
        all_user_queries.append(cur_user_query)
    if cur_agent_response:
        formatted = _get_agent_response(cur_agent_response, include_tool_messages=True)
        all_agent_responses.append([formatted])

    conversation_history: Dict = {
        "user_queries": all_user_queries,
        "agent_responses": all_agent_responses[:len(all_user_queries) - 1]
        if len(all_user_queries) > 0
        else [],
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


class GroundednessEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """
    Evaluates groundedness score.

    Takes query (optional), response, and context or a multi-turn conversation, including reasoning.

    The groundedness measure assesses the correspondence between claims in an AI-generated answer and the source
    context, making sure that these claims are substantiated by the context. Even if the responses from LLM are
    factually correct, they'll be considered ungrounded if they can't be verified against the provided sources
    (such as your input source or your database). Use the groundedness metric when you need to verify that
    AI-generated responses align with and are validated by the provided context.

    Groundedness scores range from 1 to 5, with 1 being the least grounded and 5 being the most grounded.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the groundedness evaluator. Default is 3.
    :type threshold: int
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, the evaluator will use reasoning model configuration (o1/o3 models).
        This will adjust parameters like max_completion_tokens and remove unsupported parameters. Default is False.
    :paramtype is_reasoning_model: bool

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START groundedness_evaluator]
            :end-before: [END groundedness_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a GroundednessEvaluator.

    .. admonition:: Example with Threshold:
        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_groundedness_evaluator]
            :end-before: [END threshold_groundedness_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a GroundednessEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START groundedness_evaluator]
            :end-before: [END groundedness_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call GroundednessEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE_NO_QUERY = "groundedness_without_query.prompty"
    _PROMPTY_FILE_WITH_QUERY = "groundedness_with_query.prompty"
    _MULTI_TURN_PROMPTY_FILE = "groundedness_multi_turn.prompty"
    _RESULT_KEY = "groundedness"
    _OPTIONAL_PARAMS = ["query", "messages", "tool_definitions"]
    _SUPPORTED_TOOLS = ["file_search"]

    _validator: ValidatorInterface
    _validator_with_query: ValidatorInterface
    _validator_messages: ValidatorInterface

    id = "azureai://built-in/evaluators/groundedness"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=3, credential=None, evaluation_level=None, **kwargs):
        """Initialize a GroundednessEvaluator instance.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword threshold: The threshold for the groundedness evaluator. Default is 3.
        :type threshold: int
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword evaluation_level: Force a specific evaluation level for this invocation. When ``None``
            (default), the level is auto-detected from input shape (``messages`` -> conversation,
            ``query``/``response`` -> trace). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TRACE`` to override auto-detection.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE_NO_QUERY)  # Default to no query

        self._higher_is_better = True

        # Validate and store evaluation level
        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ErrorTarget.GROUNDEDNESS_EVALUATOR
        )

        # Initialize input validators
        self._validator = ConversationValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=True
        )

        self._validator_with_query = ConversationValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR, requires_query=True,
            check_for_unsupported_tools=True
        )

        self._validator_messages = MessagesOrQueryResponseInputValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=False
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )
        self._model_config = model_config
        self.threshold = threshold
        self._credential = credential

        # Load the multi-turn prompty flow for conversation-level evaluation
        multi_turn_prompty_path = os.path.join(current_dir, self._MULTI_TURN_PROMPTY_FILE)
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(model_config),
            self._DEFAULT_OPEN_API_VERSION,
            UserAgentSingleton().value,
        )
        self._multi_turn_flow = AsyncPrompty.load(
            source=multi_turn_prompty_path,
            model=prompty_model_config,
            is_reasoning_model=self._is_reasoning_model,
            token_credential=credential,
        )

    @overload
    def __call__(
        self,
        *,
        response: str,
        context: str,
        query: Optional[str] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate groundedness for given input of response, context, and optional query.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword context: The context to be evaluated.
        :paramtype context: str
        :keyword query: The query to be evaluated. Optional parameter for use with the `response`
            and `context` parameters. If provided, a different prompt template will be used for evaluation.
        :paramtype query: Optional[str]
        :return: The groundedness score.
        :rtype: Dict[str, float]
        """

    @overload
    def __call__(
        self,
        *,
        query: str | List[dict],
        response: str | List[dict],
        tool_definitions: Optional[List[dict]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate groundedness for agent response with tool calls. Only file_search tool is supported.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response from the agent to be evaluated.
        :paramtype response: List[dict]
        :keyword tool_definitions: Optional tool definitions used by the agent.
        :paramtype tool_definitions: Optional[List[dict]]
        :return: The groundedness score.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate groundedness for a conversation.

        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The groundedness score.
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate groundedness for a full multi-turn conversation.

        Evaluates whether the agent's responses remain grounded in the provided context,
        tool results, and user-provided information throughout all turns.

        :keyword messages: The full multi-turn conversation as a list of message dicts.
        :paramtype messages: List[dict]
        :keyword tool_definitions: An optional list of tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: The groundedness score.
        :rtype: Dict[str, Union[str, float]]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Evaluate groundedness.

        Accepts either a query, response, and context for a single evaluation,
        a conversation for a multi-turn per-turn evaluation, or messages for
        conversation-level evaluation.

        :keyword query: The query to be evaluated. Optional parameter for use
            with the `response` and `context` parameters.
        :paramtype query: Optional[str]
        :keyword response: The response to be evaluated.
        :paramtype response: Optional[str]
        :keyword context: The context to be evaluated.
        :paramtype context: Optional[str]
        :keyword conversation: The conversation to evaluate per-turn.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :keyword messages: The full multi-turn conversation for conversation-level evaluation.
        :paramtype messages: Optional[List[dict]]
        :keyword tool_definitions: Optional tool definitions for conversation-level evaluation.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: The groundedness score.
        :rtype: Union[Dict[str, Union[str, float]], Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]]
        """
        if kwargs.get("query", None):
            self._ensure_query_prompty_loaded()

        return super().__call__(*args, **kwargs)

    def _ensure_query_prompty_loaded(self):
        """Switch to the query prompty file if not already loaded."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE_WITH_QUERY)

        self._prompty_file = prompty_path
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(self._model_config),
            self._DEFAULT_OPEN_API_VERSION,
            UserAgentSingleton().value,
        )
        self._flow = AsyncPrompty.load(
            source=self._prompty_file,
            model=prompty_model_config,
            is_reasoning_model=self._is_reasoning_model,
            token_credential=self._credential,
        )

    def has_context(self, eval_input: dict) -> bool:
        """
        Return True if eval_input contains a non-empty 'context' field.

        Treats None, empty strings, empty lists, and lists of empty strings as no context.
        """
        context = eval_input.get("context", None)
        return self._validate_context(context)

    def _validate_context(self, context) -> bool:
        """
        Validate if the provided context is non-empty and meaningful.

        Treats None, empty strings, empty lists, and lists of empty strings as no context.
        :param context: The context to validate
        :type context: Union[str, List, None]
        :return: True if context is valid and non-empty, False otherwise
        :rtype: bool
        """
        if not context:
            return False
        if context == "<>":  # Special marker for no context
            return False
        if isinstance(context, list):
            return any(str(c).strip() for c in context)
        if isinstance(context, str):
            return bool(context.strip())
        return True

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

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
        if self._evaluation_level == EvaluationLevel.TRACE:
            return False
        # Auto-detect (_evaluation_level is None)
        return eval_input.get("messages") is not None

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        # Route to conversation-level evaluation if appropriate
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self.threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        if eval_input.get("query", None) is None:
            result = await super()._do_eval(eval_input)
            # Check if base returned nan (invalid output case)
            if math.isnan(result.get(self._result_key, 0)):
                raise EvaluationException(
                    message="Evaluator returned invalid output.",
                    blame=ErrorBlame.SYSTEM_ERROR,
                    category=ErrorCategory.FAILED_EXECUTION,
                    target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
                )
            return result

        contains_context = self.has_context(eval_input)

        simplified_query = simplify_messages(eval_input["query"], drop_tool_calls=contains_context)
        simplified_response = simplify_messages(eval_input["response"], drop_tool_calls=False)

        # Build simplified input
        simplified_eval_input = {
            "query": simplified_query,
            "response": simplified_response,
            "context": eval_input["context"],
        }

        # Replace and call the parent method
        result = await super()._do_eval(simplified_eval_input)
        # Check if base returned nan (invalid output case)
        if math.isnan(result.get(self._result_key, 0)):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            )
        return result

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Evaluate groundedness for a full conversation-level evaluation.

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
        return self._parse_conversation_prompty_output(prompty_output_dict)

    def _parse_conversation_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[float, str]]:
        """Parse the multi-turn prompty output into a standardized result dictionary.

        :param prompty_output_dict: Raw output from the prompty flow.
        :type prompty_output_dict: Dict
        :return: The parsed evaluation result.
        :rtype: Dict
        """
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            score = llm_output.get("score", None)
            if score is None or not isinstance(score, (int, float)):
                raise EvaluationException(
                    message="Evaluator returned invalid output: missing or invalid 'score' field.",
                    blame=ErrorBlame.SYSTEM_ERROR,
                    category=ErrorCategory.FAILED_EXECUTION,
                    target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
                )
            score = int(score)
            score_result = "pass" if score >= self.threshold else "fail"
            reason = llm_output.get("reason", "")
            return {
                self._result_key: score,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_threshold": self.threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_details": llm_output.get("properties", {}),
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
        )

    async def _real_call(self, **kwargs):
        """Asynchronous call where real end-to-end evaluation logic is performed.

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
        elif self._evaluation_level == EvaluationLevel.TRACE and kwargs.get("messages"):
            if any(m.get("role") == MessageRole.USER for m in kwargs["messages"]):
                query_messages, response_messages = _split_messages_at_latest_user(kwargs["messages"])
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages

        # Validate input before processing
        if kwargs.get("messages"):
            self._validator_messages.validate_eval_input(kwargs)
        elif kwargs.get("query"):
            self._validator_with_query.validate_eval_input(kwargs)
        else:
            self._validator.validate_eval_input(kwargs)

        # Convert inputs into list of evaluable inputs.
        try:
            return await super()._real_call(**kwargs)
        except EvaluationException as ex:
            if ex.category == ErrorCategory.NOT_APPLICABLE:
                return {
                    self._result_key: self.threshold,
                    f"{self._result_key}_result": "pass",
                    f"{self._result_key}_threshold": self.threshold,
                    f"{self._result_key}_reason": f"Not applicable: {ex.message}",
                    f"{self._result_key}_details": {},
                    f"{self._result_key}_prompt_tokens": 0,
                    f"{self._result_key}_completion_tokens": 0,
                    f"{self._result_key}_total_tokens": 0,
                    f"{self._result_key}_finish_reason": "",
                    f"{self._result_key}_model": "",
                    f"{self._result_key}_sample_input": "",
                    f"{self._result_key}_sample_output": "",
                }
            else:
                raise ex

    def _is_single_entry(self, value):
        """Determine if the input value represents a single entry, unsure is returned as False."""
        if isinstance(value, str):
            return True
        if isinstance(value, list) and len(value) == 1:
            return True
        return False

    def _convert_kwargs_to_eval_input(self, **kwargs):
        if kwargs.get("messages") is not None:
            return super()._convert_kwargs_to_eval_input(**kwargs)

        if kwargs.get("context") or kwargs.get("conversation"):
            return super()._convert_kwargs_to_eval_input(**kwargs)
        query = kwargs.get("query")
        response = kwargs.get("response")
        tool_definitions = kwargs.get("tool_definitions")

        if query and self._prompty_file != self._PROMPTY_FILE_WITH_QUERY:
            self._ensure_query_prompty_loaded()

        if (not query) or (not response):  # or not tool_definitions:
            msg = (
                f"{type(self).__name__}: Either 'conversation' or individual inputs must be provided. "
                "For Agent groundedness 'query' and 'response' are required."
            )
            raise EvaluationException(
                message=msg,
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            )

        # If response is a string, we can skip the context extraction and just return the eval input
        if response and isinstance(response, str):
            return super()._convert_kwargs_to_eval_input(query=query, response=response, context=response)

        context = self._get_context_from_agent_response(response, tool_definitions)

        if not self._validate_context(context) and self._is_single_entry(response) and self._is_single_entry(query):
            msg = (
                f"{type(self).__name__}: No valid context provided or could be extracted from the query or response. "
                "Please either provide valid context or pass the full items list for 'response' and 'query' "
                "to extract context from tool calls."
            )
            raise EvaluationException(
                message=msg,
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.NOT_APPLICABLE,
                target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            )

        filtered_response = self._filter_file_search_results(response) if self._validate_context(context) else response
        return super()._convert_kwargs_to_eval_input(response=filtered_response, context=context, query=query)

    def _filter_file_search_results(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out file_search tool results from the messages."""
        file_search_ids = self._get_file_search_tool_call_ids(messages)
        return [
            msg for msg in messages if not (msg.get("role") == "tool" and msg.get("tool_call_id") in file_search_ids)
        ]

    def _get_context_from_agent_response(self, response, tool_definitions):
        """Extract context text from file_search tool results in the agent response."""
        NO_CONTEXT = "<>"
        context = ""
        try:
            logger.debug("Extracting context from response")
            tool_calls = self._parse_tools_from_response(response=response)
            logger.debug(f"Tool Calls parsed successfully: {tool_calls}")

            if not tool_calls:
                return NO_CONTEXT

            context_lines = []
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict) or tool_call.get("type") != "tool_call":
                    continue

                tool_name = tool_call.get("name")
                if tool_name != "file_search":
                    continue

                # Extract tool results
                for result in tool_call.get("tool_result", []):
                    results = result if isinstance(result, list) else [result]
                    for r in results:
                        file_name = r.get("file_name", "Unknown file name")
                        for content in r.get("content", []):
                            text = content.get("text")
                            if text:
                                context_lines.append(f"{file_name}:\n- {text}---\n\n")

            context = "\n".join(context_lines) if len(context_lines) > 0 else None

        except Exception as ex:
            logger.debug(f"Error extracting context from agent response : {str(ex)}")
            context = None

        context = context if context else NO_CONTEXT
        return context

    def _get_file_search_tool_call_ids(self, query_or_response):
        """Return a list of tool_call_ids for file search tool calls."""
        tool_calls = self._parse_tools_from_response(query_or_response)
        return [tc.get("tool_call_id") for tc in tool_calls if tc.get("name") == "file_search"]
