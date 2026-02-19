# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from itertools import chain
import os
import logging
from typing import Dict, List, Union, TypeVar
from typing_extensions import overload, override
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import (
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
    EvaluationException,
)
from azure.ai.evaluation._common.utils import check_score_is_valid
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._converters._models import (
    _BUILT_IN_DESCRIPTIONS,
    _BUILT_IN_PARAMS,
)
from enum import Enum

from abc import ABC, abstractmethod
from typing import Any, Optional


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


class ConversationValidator(ValidatorInterface):
    """Validate conversation inputs (queries and responses) comprised of message lists."""

    requires_query: bool = True
    check_for_unsupported_tools: bool = False
    error_target: ErrorTarget

    UNSUPPORTED_TOOLS: List[str] = [
        "web_search_call",
        "code_interpreter_call",
        "azure_ai_search_call",
        "bing_grounding_call",
        "bing_custom_search_preview_call",
        "azure_fabric",
        "sharepoint_grounding",
        "browser_automation",
        "openapi_call"
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
                                        f"{self.error_target} evaluator."
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

            if content_type in [ContentType.TOOL_RESULT, ContentType.OPENAPI_CALL_OUTPUT]:
                error = self._validate_field_exists(
                    content_item, "tool_result", f"content items for role '{MessageRole.TOOL.value}'"
                )
                if error:
                    return error
            elif content_type == ContentType.FUNCTION_CALL_OUTPUT:
                error = self._validate_field_exists(
                    content_item, "function_call_output", f"content items for role '{MessageRole.TOOL.value}'"
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


class ToolCallsValidator(ToolDefinitionsValidator):
    """Validate tool calls alongside tool definitions and conversation inputs."""

    optional_tool_definitions = False

    def __init__(
        self,
        error_target: ErrorTarget,
        requires_query: bool = True,
        optional_tool_definitions: bool = False,
        check_for_unsupported_tools: bool = False
    ):
        """Initialize ToolCallsValidator."""
        super().__init__(error_target, requires_query, optional_tool_definitions, check_for_unsupported_tools)

    def _validate_tool_calls(self, tool_calls) -> Optional[EvaluationException]:
        """Validate tool calls input."""
        if not tool_calls:
            return EvaluationException(
                message="No tool calls found in response or provided tool_calls.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=self.error_target,
            )
        if isinstance(tool_calls, str):
            return None
        if not isinstance(tool_calls, list):
            return EvaluationException(
                message="Tool calls must be provided as a list of dictionaries.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
        for tool_call in tool_calls:
            if not tool_call or not isinstance(tool_call, dict):
                return EvaluationException(
                    message="Each tool call must be a dictionary.",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            tool_call_validation_exception = self._validate_tool_call_content_item(tool_call)
            if tool_call_validation_exception:
                return tool_call_validation_exception
        return None

    @override
    def validate_eval_input(self, eval_input: Dict[str, Any]) -> bool:
        """Validate the evaluation input dictionary."""
        query = eval_input.get("query")
        query_validation_exception = self._validate_query(query)
        if query_validation_exception:
            raise query_validation_exception

        tool_definitions = eval_input.get("tool_definitions")
        tool_definitions_validation_exception = self._validate_tool_definitions(tool_definitions)
        if tool_definitions_validation_exception:
            raise tool_definitions_validation_exception

        response = eval_input.get("response")
        response_validation_exception = self._validate_response(response)

        tool_calls = eval_input.get("tool_calls")
        tool_calls_validation_exception = self._validate_tool_calls(tool_calls)

        if response_validation_exception and tool_calls_validation_exception:
            main_exception: EvaluationException
            if response_validation_exception.category == ErrorCategory.MISSING_FIELD:
                main_exception = tool_calls_validation_exception
                main_exception.inner_exception = response_validation_exception
            else:
                main_exception = response_validation_exception
                main_exception.inner_exception = tool_calls_validation_exception
            raise main_exception

        return True


# endregion Validators

logger = logging.getLogger(__name__)

T_EvalValue = TypeVar("T_EvalValue")


def _get_built_in_definition(tool_name: str):
    """Get the definition for the built-in tool."""
    if tool_name in _BUILT_IN_DESCRIPTIONS:
        return {
            "type": tool_name,
            "description": _BUILT_IN_DESCRIPTIONS[tool_name],
            "name": tool_name,
            "parameters": _BUILT_IN_PARAMS.get(tool_name, {}),
        }
    return None


def _get_needed_built_in_definitions(tool_calls: List[Dict]) -> List[Dict]:
    """Extract tool definitions needed for the given built-in tool calls."""
    needed_definitions = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            tool_type = tool_call.get("type")

            # Only support converter format: {type: "tool_call", name: "bing_custom_search", arguments: {...}}
            if tool_type == "tool_call":
                tool_name = tool_call.get("name")
                if tool_name in _BUILT_IN_DESCRIPTIONS:
                    built_in_def = _get_built_in_definition(tool_name)
                    if built_in_def and built_in_def not in needed_definitions:
                        needed_definitions.append(built_in_def)

    return needed_definitions


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
    """Normalize function_call/function_call_output types to tool_call/tool_result."""
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
    return messages


def _preprocess_messages(messages):
    """Drop MCP approval messages and normalize function call types."""
    messages = _drop_mcp_approval_messages(messages)
    messages = _normalize_function_call_types(messages)
    return messages


@experimental
class ToolCallAccuracyEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Tool Call Accuracy evaluator assesses how accurately an AI uses tools by examining the following criteria.

        - Relevance to the conversation.
        - Parameter correctness according to tool definitions.
        - Parameter value extraction from the conversation.

    The evaluator uses a scoring rubric of 1 to 5:
        - Score 1: The tool calls are irrelevant
        - Score 2: The tool calls are partially relevant, but not enough tools were called or the
          parameters were not correctly passed.
        - Score 3: The tool calls are relevant, but there were unnecessary, excessive tool calls made.
        - Score 4: The tool calls are relevant, but some tools returned errors and agent retried
          calling them again and succeeded.
        - Score 5: The tool calls are relevant, and all parameters were correctly passed.

    This evaluation focuses on measuring whether tool calls meaningfully contribute to addressing
    user needs while properly following tool definitions and using information present in the
    conversation history.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START tool_call_accuracy_evaluator]
            :end-before: [END tool_call_accuracy_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolCallAccuracyEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START tool_call_accuracy_evaluator]
            :end-before: [END tool_call_accuracy_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call ToolCallAccuracyEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "tool_call_accuracy.prompty"
    _RESULT_KEY = "tool_call_accuracy"

    _MAX_TOOL_CALL_ACCURACY_SCORE = 5
    _MIN_TOOL_CALL_ACCURACY_SCORE = 1
    _DEFAULT_TOOL_CALL_ACCURACY_SCORE = 3

    _NO_TOOL_CALLS_MESSAGE = "No tool calls found in response or provided tool_calls."
    _NO_TOOL_DEFINITIONS_MESSAGE = "Tool definitions must be provided."
    _TOOL_DEFINITIONS_MISSING_MESSAGE = "Tool definitions for all tool calls must be provided."
    _INVALID_SCORE_MESSAGE = "Tool call accuracy score must be between 1 and 5."

    _LLM_SCORE_KEY = "tool_calls_success_level"

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/tool_call_accuracy"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=_DEFAULT_TOOL_CALL_ACCURACY_SCORE, credential=None, **kwargs):
        """Initialize the Tool Call Accuracy evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: int
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", threshold)
        self.threshold = threshold_value

        # Initialize input validator
        self._validator = ToolCallsValidator(
            error_target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
            check_for_unsupported_tools=True,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        tool_definitions: Union[dict, List[dict]],
        tool_calls: Union[dict, List[dict]] = None,
        response: Union[str, List[dict]] = None,
    ) -> Dict[str, Union[str, float]]:
        """
        Evaluate tool call accuracy. Accepts a query, tool definitions, and tool calls for evaluation.

        :keyword query: Query or Chat history up to the message that has the tool call being evaluated.
        :paramtype query: Union[str, List[dict]]
        :keyword tool_definitions: List of tool definitions whose calls are being evaluated.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :keyword tool_calls: Optional List of tool calls to evaluate. If not provided response should be provided
            and should have tool call(s) in it.
        :paramtype tool_calls: Union[dict, List[dict]]
        :keyword response: Optional response to be evaluated alongside the tool calls.
            If provided all tool calls in response will be evaluated when tool_calls parameter is not provided.
            If provided and tool_calls parameter is provided, only the tool calls in tool_calls parameter
            will be evaluated.
                If response has extra tool calls they will not be evaluated, response will be used to extract
                any tool calls that are needed for evaluating a certain tool call.
            Recommended to provide it when there are tool calls that depend on output of a previous tool call.
        :paramtype response: Union[str, List[dict]]
        :return: The tool selection evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    def _convert_kwargs_to_eval_input(self, **kwargs):
        """Convert an arbitrary input into a list of inputs for evaluators.

        It is assumed that evaluators generally make use of their inputs in one of two ways.
        Either they receive a collection of keyname inputs that are all single values
        (like a query and response), or they receive conversation that iss a list of dictionary
        values.

        The self._singleton_inputs list assigned during initialization is used to find and extract
        singleton keywords, and self._allow_conversation_input is used to determine if a conversation
        is a valid input.

        If both conversations and singletons are allowed, the function will raise an exception if both
        are inputted.

        This function must be overridden by child classes IF they need to both a conversation and
        other inputs to be passed in.

        :keyword kwargs: The inputs to convert.
        :type kwargs: Dict
        :return: A list of arbitrary values that are valid inputs for this evaluator's do_eval function.
        :rtype: List
        """
        # TODO add warning that only tool calls of type function are supported
        # Collect inputs
        tool_calls = kwargs.get("tool_calls")
        tool_definitions = kwargs.get("tool_definitions", [])  # Default to empty list
        query = kwargs.get("query")
        response = kwargs.get("response")
        # TODO : Support classes that represents tool calls, messages etc once client side definitions are available
        if response:
            parsed_tool_calls = self._parse_tools_from_response(response)
            if parsed_tool_calls:
                tool_calls = parsed_tool_calls

        if not tool_calls:
            # If no tool calls provided and response is string, use response string as tool calls as is
            if response and isinstance(response, str):
                tool_calls = response
            else:
                return {"error_message": self._NO_TOOL_CALLS_MESSAGE}

        if not isinstance(tool_calls, list) and not isinstance(tool_calls, str):
            tool_calls = [tool_calls]
        if not isinstance(tool_definitions, list) and not isinstance(tool_definitions, str):
            tool_definitions = [tool_definitions] if tool_definitions else []

        if isinstance(tool_calls, str) or isinstance(tool_definitions, str):
            needed_tool_definitions = tool_definitions
        else:
            try:
                needed_tool_definitions = self._extract_needed_tool_definitions(tool_calls, tool_definitions)
            except EvaluationException:
                # Check if this is because no tool definitions were provided at all
                if len(tool_definitions) == 0:
                    return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}
                else:
                    return {"error_message": self._TOOL_DEFINITIONS_MISSING_MESSAGE}

        if not needed_tool_definitions:
            return {"error_message": self._NO_TOOL_DEFINITIONS_MESSAGE}

        return {
            "query": query,
            "tool_calls": tool_calls,
            "tool_definitions": needed_tool_definitions,
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a tool call accuracy evaluation.

        :param eval_input: The input to the evaluator. Expected to contain
        whatever inputs are needed for the _flow method, including context
        and other fields depending on the child class.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if eval_input.get("query") is None:
            raise EvaluationException(
                message=(
                    "Query is a required input to the Tool Call Accuracy evaluator."
                ),
                internal_message=(
                    "Query is a required input to the Tool Call Accuracy evaluator."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
            )

        # Single LLM call for all tool calls
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            score = llm_output.get(self._LLM_SCORE_KEY, None)
            if not score or not check_score_is_valid(
                score,
                ToolCallAccuracyEvaluator._MIN_TOOL_CALL_ACCURACY_SCORE,
                ToolCallAccuracyEvaluator._MAX_TOOL_CALL_ACCURACY_SCORE,
            ):
                raise EvaluationException(
                    message=f"Invalid score value: {score}. Expected a number in range "
                            f"[{ToolCallAccuracyEvaluator._MIN_TOOL_CALL_ACCURACY_SCORE}, "
                            f"{ToolCallAccuracyEvaluator._MAX_TOOL_CALL_ACCURACY_SCORE}].",
                    internal_message="Invalid score value.",
                    category=ErrorCategory.FAILED_EXECUTION,
                    blame=ErrorBlame.SYSTEM_ERROR,
                )

            # Format the output
            reason = llm_output.get("chain_of_thought", "")
            score = float(score)
            score_result = "pass" if score >= self.threshold else "fail"
            response_dict = {
                self._result_key: score,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_details": llm_output.get("details", {}),
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
            return response_dict

        else:
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
            )

    async def _real_call(self, **kwargs):
        """Perform the asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
        self._validator.validate_eval_input(kwargs)

        response = kwargs.get("response")
        if _is_intermediate_response(response):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self.threshold,
            )
        if "response" in kwargs:
            kwargs["response"] = _preprocess_messages(kwargs["response"])
        if "query" in kwargs:
            kwargs["query"] = _preprocess_messages(kwargs["query"])
        # Convert inputs into list of evaluable inputs.
        eval_input = self._convert_kwargs_to_eval_input(**kwargs)
        if isinstance(eval_input, dict) and eval_input.get("error_message"):
            # If there is an error message, return not applicable result
            return self._not_applicable_result(eval_input.get("error_message"), self.threshold)
        # Do the evaluation
        result = await self._do_eval(eval_input)
        # Return the result
        return result

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the tool call is not applicable for evaluation.

        :param error_message: The error message indicating why the evaluation is not applicable.
        :type error_message: str
        :param threshold: The threshold value for the evaluation.
        :type threshold: Union[int, float]
        :return: A dictionary containing the result of the evaluation.
        :rtype: Dict[str, Union[str, float]]
        """
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_details": {},
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

    def _extract_needed_tool_definitions(self, tool_calls, tool_definitions):
        """Extract the tool definitions that are needed for the provided tool calls."""
        needed_tool_definitions = []

        # Add all user-provided tool definitions
        needed_tool_definitions.extend(tool_definitions)

        # Add the needed built-in tool definitions (if they are called)
        built_in_definitions = _get_needed_built_in_definitions(tool_calls)
        needed_tool_definitions.extend(built_in_definitions)

        # OpenAPI tool is a collection of functions, so we need to expand it
        tool_definitions_expanded = list(
            chain.from_iterable(
                tool.get("functions", []) if tool.get("type") == "openapi" else [tool]
                for tool in needed_tool_definitions
            )
        )

        # Validate that all tool calls have corresponding definitions
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                tool_type = tool_call.get("type")

                if tool_type == "tool_call":
                    tool_name = tool_call.get("name")
                    if tool_name and tool_name in _BUILT_IN_DESCRIPTIONS:
                        # This is a built-in tool from converter, already handled above
                        continue
                    elif tool_name:
                        # This is a regular function tool from converter or built-in tool from agent v2
                        tool_definition_exists = any(
                            tool.get("name") == tool_name for tool in tool_definitions_expanded
                        )
                        if not tool_definition_exists:
                            raise EvaluationException(
                                message=f"Tool definition for {tool_name} not found",
                                blame=ErrorBlame.USER_ERROR,
                                category=ErrorCategory.INVALID_VALUE,
                                target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
                            )
                    else:
                        raise EvaluationException(
                            message=f"Tool call missing name: {tool_call}",
                            blame=ErrorBlame.USER_ERROR,
                            category=ErrorCategory.INVALID_VALUE,
                            target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
                        )
                else:
                    # Unsupported tool format - only converter format is supported
                    raise EvaluationException(
                        message=f"Unsupported tool call format. Only converter format is supported: {tool_call}",
                        blame=ErrorBlame.USER_ERROR,
                        category=ErrorCategory.INVALID_VALUE,
                        target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
                    )
            else:
                # Tool call is not a dictionary
                raise EvaluationException(
                    message=f"Tool call is not a dictionary: {tool_call}",
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=ErrorTarget.TOOL_CALL_ACCURACY_EVALUATOR,
                )

        return needed_tool_definitions

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate tool call accuracy. Accepts a query, tool definitions, and tool calls for evaluation.

        :keyword query: Query or Chat history up to the message that has the tool call being evaluated.
        :paramtype query: Union[str, List[dict]]
        :keyword tool_definitions: List of tool definitions whose calls are being evaluated.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :keyword tool_calls: Optional List of tool calls to evaluate. If not provided response should be provided
            and should have tool call(s) in it.
        :paramtype tool_calls: Union[dict, List[dict]]
        :keyword response: Optional response to be evaluated alongside the tool calls.
            If provided all tool calls in response will be evaluated when tool_calls parameter is not provided.
            If provided and tool_calls parameter is provided, only the tool calls in tool_calls parameter
            will be evaluated.
                If response has extra tool calls they will not be evaluated, response will be used to extract
                any tool calls that are needed for evaluating a certain tool call.
            Recommended to provide it when there are tool calls that depend on output of a previous tool call.
        :paramtype response: Union[str, List[dict]]
        :return: The tool selection evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """
        return super().__call__(*args, **kwargs)
