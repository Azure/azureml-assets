# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import os
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional, Union, List, Tuple

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    validate_model_config,
    _extract_text_from_content,
    _get_agent_response,
    _pretty_format_conversation_history,
)


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
    DEVELOPER = "developer"


class EvaluationLevel(str, Enum):
    """Supported evaluation levels for CoherenceEvaluator.

    - ``CONVERSATION``: Force conversation-level evaluation using the multi-turn path.
    - ``TURN``: Force turn-level evaluation using the single-turn query/response path.
    """

    CONVERSATION = "conversation"
    TURN = "turn"


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


def _resolve_evaluation_level(
    evaluation_level: Optional[Union[EvaluationLevel, str]],
    error_target: ErrorTarget,
) -> Optional[EvaluationLevel]:
    """Validate and normalize the evaluation_level parameter."""
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
                    f"The 'content' field must be a list of dictionaries messages "
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
        """Validate the evaluation input dictionary.

        Supports two input modes:
        1. ``conversation`` — dict with a ``messages`` key (legacy conversation format)
        2. ``query`` / ``response`` — single-turn evaluation
        """
        # Legacy conversation path
        conversation = eval_input.get("conversation")
        if conversation:
            conversation_validation_exception = self._validate_conversation(conversation)
            if conversation_validation_exception:
                raise conversation_validation_exception
            return True

        # Single-turn query/response path
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
    """Serialize a list of chat messages into a labeled transcript for multi-turn coherence."""
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

        if role in (MessageRole.SYSTEM, MessageRole.DEVELOPER):
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


logger = logging.getLogger(__name__)


class CoherenceEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """
    Evaluates coherence score for a given query and response or a multi-turn conversation, including reasoning.

    The coherence measure assesses the ability of the language model to generate text that reads naturally,
    flows smoothly, and resembles human-like language in its responses. Use it when assessing the readability
    and user-friendliness of a model's generated responses in real-world applications.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the coherence evaluator. Default is 3.
    :type threshold: int
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, the evaluator will use reasoning model configuration (o1/o3 models).
        This will adjust parameters like max_completion_tokens and remove unsupported parameters. Default is False.
    :paramtype is_reasoning_model: bool

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START coherence_evaluator]
            :end-before: [END coherence_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call CoherenceEvaluator using azure.ai.evaluation.AzureAIProject

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START coherence_evaluator]
            :end-before: [END coherence_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call CoherenceEvaluator using Azure AI Project URL in following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_coherence_evaluator]
            :end-before: [END threshold_coherence_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a CoherenceEvaluator with a query and response.

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "coherence.prompty"
    _MULTI_TURN_PROMPTY_FILE = "coherence_multi_turn.prompty"
    _RESULT_KEY = "coherence"
    _OPTIONAL_PARAMS = ["messages"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/coherence"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=3, credential=None, evaluation_level=None, **kwargs):
        """Initialize the Coherence evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: int
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        :keyword evaluation_level: Force a specific evaluation level for this invocation. When ``None``
            (default), the level is auto-detected from input shape (``messages`` -> conversation,
            ``query``/``response`` -> turn). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TURN`` to override auto-detection.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True

        # Validate and store evaluation level
        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ErrorTarget.COHERENCE_EVALUATOR
        )

        # Initialize input validator (supports both query/response and messages)
        self._validator = MessagesOrQueryResponseInputValidator(error_target=ErrorTarget.COHERENCE_EVALUATOR)

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

        # Load the multi-turn prompty flow for conversation-level evaluation
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
        query: str,
        response: str,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate coherence for given input of query, response.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :return: The coherence score.
        :rtype: Dict[str, float]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate coherence for a conversation.

        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The coherence score.
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate coherence for a full multi-turn conversation."""

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Evaluate coherence.

        Accepts either a query and response for a single evaluation,
        or a conversation for a potentially multi-turn evaluation. If the conversation has more than one pair of
        turns, the evaluator will aggregate the results of each turn.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: Optional[str]
        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages". Conversation turns are expected
            to be dictionaries with keys "content" and "role".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The relevance score.
        :rtype: Union[Dict[str, float], Dict[str, Union[float, Dict[str, List[float]]]]]
        """
        return super().__call__(*args, **kwargs)

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_properties": {},
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

    def _should_use_conversation_level(self, eval_input: Dict) -> bool:
        """Determine whether to use conversation-level evaluation."""
        if self._evaluation_level == EvaluationLevel.CONVERSATION:
            return True
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
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
        """Build a standardized result dictionary for multi-turn coherence outputs."""
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        return {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_properties": properties,
            f"{self._result_key}_prompt_tokens": p.get("input_token_count", 0),
            f"{self._result_key}_completion_tokens": p.get("output_token_count", 0),
            f"{self._result_key}_total_tokens": p.get("total_token_count", 0),
            f"{self._result_key}_finish_reason": p.get("finish_reason", ""),
            f"{self._result_key}_model": p.get("model_id", ""),
            f"{self._result_key}_sample_input": p.get("sample_input", ""),
            f"{self._result_key}_sample_output": p.get("sample_output", ""),
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

        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a coherence evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        eval_input.pop("messages", None)

        result = await super()._do_eval(eval_input)

        # Check if base returned nan (invalid output case)
        if math.isnan(result.get(self._result_key, 0)):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.COHERENCE_EVALUATOR,
            )
        return result

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Evaluate coherence for a full multi-turn conversation."""
        messages = _preprocess_messages(eval_input["messages"])
        conversation_text = serialize_messages(messages)
        prompty_output_dict = await self._multi_turn_flow(
            timeout=self._LLM_CALL_TIMEOUT,
            messages=conversation_text,
        )
        return self._parse_prompty_output(prompty_output_dict)

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Parse multi-turn prompty JSON output into evaluator result schema."""
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
        score = None
        result = "error"
        reason = "Evaluator returned invalid output."
        status = "error"
        properties = {}

        if isinstance(llm_output, dict):
            status = str(llm_output.get("status", "completed")).strip().lower()
            reason = llm_output.get("reason", "")
            properties = llm_output.get("properties") or {}

            if status == "skipped":
                result = "not_applicable"
                reason = reason or "Conversation coherence cannot be evaluated due to non-logical user flow."
            else:
                score_value = llm_output.get("score")
                if score_value is None:
                    result = "error"
                    reason = "Evaluator returned invalid output: missing 'score'."
                    status = "error"
                else:
                    try:
                        score_float = float(score_value)
                    except (TypeError, ValueError):
                        result = "error"
                        reason = f"Evaluator returned invalid output: invalid 'score' value: {score_value}"
                        status = "error"
                    else:
                        score = max(1, min(5, int(round(score_float))))
                        result = "pass" if score >= self._threshold else "fail"

        return self._build_result(
            score=score,
            result=result,
            reason=reason,
            status=status,
            properties=properties,
            prompty_output_dict=prompty_output_dict,
        )
