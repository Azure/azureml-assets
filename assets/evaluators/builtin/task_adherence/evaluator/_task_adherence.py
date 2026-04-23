# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    _extract_text_from_content,
    _get_agent_response,
    _pretty_format_conversation_history,
    construct_prompty_model_config,
    reformat_conversation_history,
    reformat_agent_response,
    reformat_tool_definitions,
    validate_model_config,
)
from azure.ai.evaluation._common._experimental import experimental

from abc import ABC, abstractmethod
from enum import Enum

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


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
    """Supported evaluation levels for TaskAdherenceEvaluator."""

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
        except ValueError as exc:
            raise EvaluationException(
                message=(
                    f"Invalid evaluation_level '{evaluation_level}'. "
                    f"Must be one of: {valid}."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=error_target,
            ) from exc
    raise EvaluationException(
        message=(
            f"Invalid evaluation_level '{evaluation_level}'. "
            f"Must be one of: {valid}."
        ),
        blame=ErrorBlame.USER_ERROR,
        category=ErrorCategory.INVALID_VALUE,
        target=error_target,
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
        """Initialize ConversationValidator."""
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
            if not all("type" in item for item in content):
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
        if not self.requires_query:
            return None
        return self._validate_input_messages_list(query, "Query")

    def _validate_response(self, response: Any) -> Optional[EvaluationException]:
        return self._validate_input_messages_list(response, "Response")

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate evaluation input."""
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


class ToolDefinitionsValidator(ConversationValidator):
    """Validate tool definitions alongside conversation inputs."""

    optional_tool_definitions: bool = True

    def __init__(
        self,
        error_target: ErrorTarget,
        requires_query: bool = True,
        optional_tool_definitions: bool = True,
        check_for_unsupported_tools: bool = False
    ):
        """Initialize ToolDefinitionsValidator."""
        super().__init__(error_target, requires_query, check_for_unsupported_tools)
        self.optional_tool_definitions = optional_tool_definitions

    def _validate_tool_definition(self, tool_definition) -> Optional[EvaluationException]:
        if not isinstance(tool_definition, dict):
            return EvaluationException(
                message="Each tool definition must be a dictionary.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        error = self._validate_string_field(tool_definition, "name", "tool definitions")
        if error:
            return error
        error = self._validate_dict_field(tool_definition, "parameters", "tool definitions")
        if error:
            return error
        return None

    def _validate_tool_definitions(self, tool_definitions) -> Optional[EvaluationException]:
        if not tool_definitions:
            if not self.optional_tool_definitions:
                return EvaluationException(
                    message="Tool definitions input is required but not provided.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.MISSING_FIELD,
                    target=self.error_target,
                )
            else:
                return None
        if isinstance(tool_definitions, str):
            return None
        if not isinstance(tool_definitions, list):
            return EvaluationException(
                message="Tool definitions must be provided as a list of dictionaries.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        for tool_definition in tool_definitions:
            if not isinstance(tool_definition, dict):
                return EvaluationException(
                    message="Each tool definition must be a dictionary.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            if tool_definition and tool_definition.get("type") == "openapi":
                error = self._validate_list_field(tool_definition, "functions", "openapi tool definition")
                if error:
                    return error
                functions_tool_definitions = tool_definition.get("functions", [])
                for function_tool_definition in functions_tool_definitions:
                    error = self._validate_tool_definition(function_tool_definition)
                    if error:
                        return error
            else:
                error = self._validate_tool_definition(tool_definition)
                if error:
                    return error
        return None

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate evaluation input with tool definitions."""
        if super().validate_eval_input(eval_input):
            tool_definitions = eval_input.get("tool_definitions")
            tool_definitions_validation_exception = self._validate_tool_definitions(tool_definitions)
            if tool_definitions_validation_exception:
                raise tool_definitions_validation_exception
        return True


class MessagesOrQueryResponseInputValidator(ToolDefinitionsValidator):
    """Validator that supports both single-turn (query/response) and multi-turn (messages) inputs."""

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

            valid_roles = {role.value for role in MessageRole}
            roles_present = set()
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise EvaluationException(
                        message=(
                            f"Each item in 'messages' must be a dictionary, "
                            f"but item at index {index} is {type(message).__name__}."
                        ),
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=self.error_target,
                    )
                role = message.get("role")
                if role is None:
                    raise EvaluationException(
                        message=f"Each message must contain a 'role' key, but message at index {index} is missing it.",
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

            last_content = messages[-1].get("content", "")
            if isinstance(last_content, list):
                has_text = any(
                    (
                        isinstance(content_item, dict)
                        and content_item.get("type") in (
                            ContentType.TEXT,
                            ContentType.INPUT_TEXT,
                            ContentType.OUTPUT_TEXT,
                        )
                    )
                    or isinstance(content_item, str)
                    for content_item in last_content
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

            for message in messages:
                error = self._validate_message_dict(message)
                if error:
                    raise error

            tool_definitions = eval_input.get("tool_definitions")
            tool_definitions_validation_exception = self._validate_tool_definitions(tool_definitions)
            if tool_definitions_validation_exception:
                raise tool_definitions_validation_exception
            return True

        return super().validate_eval_input(eval_input)


# endregion Validators


def serialize_messages(messages: List[dict]) -> str:
    """Serialize chat messages into a labeled transcript for the multi-turn prompty."""
    if not messages:
        return ""

    all_user_queries: List = []
    all_agent_responses: List = []
    current_user_query: List = []
    current_agent_response: List = []
    system_message = None

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not role:
            continue

        normalized_message = message
        if role == MessageRole.ASSISTANT and isinstance(message.get("content"), str):
            normalized_message = {**message, "content": [{"type": "text", "text": message["content"]}]}

        if role == MessageRole.SYSTEM:
            system_message = message.get("content", "")
        elif role == MessageRole.USER and "content" in message:
            if current_agent_response:
                formatted = _get_agent_response(current_agent_response, include_tool_messages=True)
                all_agent_responses.append([formatted])
                current_agent_response = []
            content = message["content"]
            if isinstance(content, str):
                text_in_message = [content]
            else:
                text_in_message = _extract_text_from_content(content)
            if text_in_message:
                current_user_query.append(text_in_message)
        elif role in (MessageRole.ASSISTANT, MessageRole.TOOL):
            if current_user_query:
                all_user_queries.append(current_user_query)
                current_user_query = []
            current_agent_response.append(normalized_message)

    if current_user_query:
        all_user_queries.append(current_user_query)
    if current_agent_response:
        formatted = _get_agent_response(current_agent_response, include_tool_messages=True)
        all_agent_responses.append([formatted])

    conversation_history: Dict[str, Any] = {
        "user_queries": all_user_queries,
        "agent_responses": all_agent_responses[: len(all_user_queries) - 1] if len(all_user_queries) > 0 else [],
    }
    if system_message:
        conversation_history["system_message"] = system_message

    result = _pretty_format_conversation_history(conversation_history)

    start_index = max(len(all_user_queries) - 1, 0)
    for index, agent_response in enumerate(all_agent_responses[start_index:], start=start_index):
        result += f"Agent turn {index + 1}:\n"
        for message_text in agent_response:
            if isinstance(message_text, list):
                for sub_message in message_text:
                    result += "  " + "\n  ".join(sub_message.split("\n")) + "\n"
            else:
                result += "  " + "\n  ".join(message_text.split("\n")) + "\n"
        result += "\n"

    return result.rstrip("\n")


logger = logging.getLogger(__name__)


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


@experimental
class TaskAdherenceEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Task Adherence evaluator assesses whether an AI assistant's actions fully align with the user's intent.

    The evaluator fully achieves the intended goal across three dimensions:

        - Goal adherence: Did the assistant achieve the user's objective within scope and constraints?
        - Rule adherence: Did the assistant respect safety, privacy, authorization, and presentation contracts?
        - Procedural adherence: Did the assistant follow required workflows, tool use, sequencing, and verification?

    The evaluator returns a boolean flag indicating whether there was any material failure in any dimension.
    A material failure is an issue that makes the output unusable, creates verifiable risk, violates an explicit
    constraint, or is a critical issue as defined in the evaluation dimensions.

    The evaluation includes step-by-step reasoning and a flagged boolean result.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START task_adherence_evaluator]
            :end-before: [END task_adherence_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call an TaskAdherenceEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START task_adherence_evaluator]
            :end-before: [END task_adherence_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call TaskAdherenceEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "task_adherence.prompty"
    _MULTI_TURN_PROMPTY_FILE = "task_adherence_multi_turn.prompty"
    _RESULT_KEY = "task_adherence"
    _OPTIONAL_PARAMS = ["tool_definitions", "messages"]

    _DEFAULT_TASK_ADHERENCE_SCORE = 0

    _validator: ValidatorInterface
    _evaluation_level: Optional[EvaluationLevel]
    _multi_turn_flow: AsyncPrompty

    id = "azureai://built-in/evaluators/task_adherence"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(
        self,
        model_config,
        *,
        threshold=_DEFAULT_TASK_ADHERENCE_SCORE,
        credential=None,
        evaluation_level=None,
        **kwargs,
    ):
        """Initialize the TaskAdherenceEvaluator.

        :param model_config: Configuration for the model
        :param threshold: Threshold for evaluation scoring
        :param credential: Authentication credential
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", threshold)
        higher_is_better_value = kwargs.pop("_higher_is_better", True)
        self.threshold = threshold_value  # to be removed in favor of _threshold

        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ErrorTarget.TASK_ADHERENCE_EVALUATOR
        )

        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=ErrorTarget.TASK_ADHERENCE_EVALUATOR
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold_value,
            credential=credential,
            _higher_is_better=higher_is_better_value,
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
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate task adherence for a given query and response.

        The query and response must be lists of messages in conversation format.


        Example with list of messages:
            evaluator = TaskAdherenceEvaluator(model_config)
            query = [
                {'role': 'system', 'content': 'You are a friendly and helpful customer service agent.'},
                {
                    'createdAt': 1700000060,
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': ('Hi, I need help with the last 2 orders on my account #888. '
                                   'Could you please update me on their status?')
                        }
                    ]
                }
            ]
            response = [
                {
                    'createdAt': 1700000070,
                    'run_id': '0',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'Hello! Let me quickly look up your account details.'}]
                },
                {
                    'createdAt': 1700000075,
                    'run_id': '0',
                    'role': 'assistant',
                    'content': [
                        {
                            'type': 'tool_call',
                            'tool_call': {
                                'id': 'tool_call_20250310_001',
                                'type': 'function',
                                'function': {
                                    'name': 'get_orders',
                                    'arguments': {'account_number': '888'}
                                }
                            }
                        }
                    ]
                },
                # ... additional response messages would continue here ...
            ]

            result = evaluator(query=query, response=response)

        :keyword query: The query being evaluated, must be a list of messages
            including system and user messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, must be a list of messages
            (full agent response including tool calls and results)
        :paramtype response: Union[str, List[dict]]
        :return: A dictionary with the task adherence evaluation results
            including flagged (bool) and reasoning (str).
        :rtype: Dict[str, Union[str, float, bool]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate task adherence for a full multi-turn conversation."""

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

    def _build_result(
        self,
        score: Optional[Union[int, float]],
        result: str,
        reason: str,
        properties: Dict[str, Any],
        *,
        threshold: Optional[Union[int, float]] = None,
        prompty_output_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[str, int, float, Dict[str, Any], None]]:
        """Build a standardized task adherence result dictionary."""
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        resolved_threshold = threshold if threshold is not None else self._threshold
        return {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_threshold": resolved_threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_details": properties,
            f"{self._result_key}_properties": properties,
            f"{self._result_key}_prompt_tokens": p.get("input_token_count", 0),
            f"{self._result_key}_completion_tokens": p.get("output_token_count", 0),
            f"{self._result_key}_total_tokens": p.get("total_token_count", 0),
            f"{self._result_key}_finish_reason": p.get("finish_reason", ""),
            f"{self._result_key}_model": p.get("model_id", ""),
            f"{self._result_key}_sample_input": p.get("sample_input", ""),
            f"{self._result_key}_sample_output": p.get("sample_output", ""),
        }

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return self._build_result(
            score=threshold,
            result="not_applicable",
            reason=f"Not applicable: {error_message}",
            properties={},
            threshold=threshold,
        )

    def _should_use_conversation_level(self, eval_input: Dict[str, Any]) -> bool:
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
        if self._evaluation_level == EvaluationLevel.CONVERSATION and not kwargs.get("messages"):
            query = kwargs.get("query")
            response = kwargs.get("response")
            if isinstance(query, str) and isinstance(response, str) and query and response:
                query, response = _wrap_string_messages(query, response)
            if isinstance(query, list) and isinstance(response, list):
                kwargs["messages"] = _merge_query_response_messages(query, response)
        elif self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            if any(message.get("role") == MessageRole.USER for message in kwargs["messages"]):
                query_messages, response_messages = _split_messages_at_latest_user(kwargs["messages"])
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages

        self._validator.validate_eval_input(kwargs)

        return await super()._real_call(**kwargs)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str, bool]]:  # type: ignore[override]
        """Do Task Adherence evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever
            inputs are needed for the _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation(eval_input)
        if "query" not in eval_input or "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Task Adherence evaluator.",
                internal_message="Both query and response must be provided as input to the Task Adherence evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.TASK_ADHERENCE_EVALUATOR,
            )

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        # Reformat conversation history and extract system message
        query_messages = reformat_conversation_history(eval_input["query"], logger, include_system_messages=True)
        system_message = ""
        user_query = ""

        # Parse query messages to extract system message and user query
        if isinstance(query_messages, list):
            for msg in query_messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_message = msg.get("content", "")
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    user_query = msg.get("content", "")
        elif isinstance(query_messages, str):
            user_query = query_messages

        # Reformat response and separate assistant messages from tool calls
        response_messages = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)
        assistant_response = ""
        tool_calls = ""

        # Parse response messages to extract assistant response and tool calls
        if isinstance(response_messages, list):
            assistant_parts = []
            tool_parts = []
            for msg in response_messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    if role == "assistant":
                        content = msg.get("content", "")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type", None) in ("text", "input_text", "output_text"):
                                        assistant_parts.append(item.get("text", ""))
                                    elif item.get("type") == "tool_call":
                                        tool_parts.append(str(item.get("tool_call", "")))
                        else:
                            assistant_parts.append(str(content))
                    elif role == "tool":
                        tool_parts.append(str(msg))
            assistant_response = "\n".join(assistant_parts)
            tool_calls = "\n".join(tool_parts)
        elif isinstance(response_messages, str):
            assistant_response = response_messages

        # Prepare inputs for prompty
        prompty_input = {
            "system_message": system_message,
            "query": user_query,
            "response": assistant_response,
            "tool_calls": tool_calls,
        }

        prompty_input.pop("messages", None)
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation(self, eval_input: Dict[str, Any]) -> Dict[str, Union[float, str, bool]]:
        """Evaluate task adherence across a full conversation."""
        messages = _preprocess_messages(eval_input["messages"])
        conversation_text = serialize_messages(messages)

        prompty_kwargs: Dict[str, Any] = {"messages": conversation_text}
        tool_definitions = eval_input.get("tool_definitions")
        if tool_definitions:
            prompty_kwargs["tool_definitions"] = reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._multi_turn_flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_kwargs)
        return self._parse_prompty_output(prompty_output_dict)

    def _parse_prompty_output(self, prompty_output_dict: Dict[str, Any]) -> Dict[str, Union[float, str, bool]]:
        """Parse prompty output into the task adherence result shape."""
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.TASK_ADHERENCE_EVALUATOR,
            )

        flagged = llm_output.get("flagged", False)
        reasoning = llm_output.get("reasoning", llm_output.get("reason", ""))
        # Convert flagged to numeric score for backward compatibility (1 = pass, 0 = fail)
        score = 0.0 if flagged else 1.0
        score_result = "fail" if flagged else "pass"
        properties = llm_output.get("details", llm_output.get("properties", {}))

        return self._build_result(
            score=score,
            result=score_result,
            reason=reasoning,
            properties=properties,
            prompty_output_dict=prompty_output_dict,
        )
