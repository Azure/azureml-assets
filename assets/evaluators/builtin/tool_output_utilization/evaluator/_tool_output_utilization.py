# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
from enum import Enum
from typing import Dict, Union, List

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import (
    EvaluationException,
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
    ErrorMessage,
)
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import _extract_text_from_content
from azure.ai.evaluation._common._experimental import experimental

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

        valid_assistant_content_types = [
            ContentType.TEXT,
            ContentType.OUTPUT_TEXT,
            ContentType.TOOL_CALL,
            ContentType.FUNCTION_CALL,
            ContentType.MCP_APPROVAL_REQUEST,
            ContentType.OPENAPI_CALL
        ]
        valid_assistant_content_types_as_strings = [t.value for t in valid_assistant_content_types]
        if isinstance(content, list):
            for content_item in content:
                content_type = content_item["type"]
                if content_type not in valid_assistant_content_types:
                    return EvaluationException(
                        message=(
                            f"Invalid content type '{content_type}' for message with "
                            f"role '{MessageRole.ASSISTANT.value}'. Must be one of "
                            f"{valid_assistant_content_types_as_strings}."
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
        valid_tool_content_types = [
            ContentType.TOOL_RESULT,
            ContentType.FUNCTION_CALL_OUTPUT,
            ContentType.MCP_APPROVAL_RESPONSE,
            ContentType.OPENAPI_CALL_OUTPUT
        ]
        valid_tool_content_types_as_strings = [t.value for t in valid_tool_content_types]
        for content_item in content:
            content_type = content_item["type"]
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

            if content_type in [ContentType.TOOL_RESULT, ContentType.OPENAPI_CALL_OUTPUT, ContentType.FUNCTION_CALL_OUTPUT]:
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


# endregion Validators

logger = logging.getLogger(__name__)


# ``` updated _exceptions.py
# Extend ErrorTarget enum if needed
def _create_extended_error_target(ErrorTarget):
    """Create an extended ErrorTarget enum that includes TOOL_OUTPUT_UTILIZATION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_OUTPUT_UTILIZATION_EVALUATOR"] = "ToolOutputUtilizationEvaluator"

    ErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ErrorTarget


ErrorTarget = _create_extended_error_target(ErrorTarget)
# ```


# ``` updated utils.py
def _filter_to_used_tools(tool_definitions, msgs_lists, logger=None):
    """Filter the tool definitions to only include those that were actually used in the messages lists."""
    try:
        used_tool_names = set()
        any_tools_used = False
        for msgs in msgs_lists:
            for msg in msgs:
                if msg.get("role") == "assistant" and "content" in msg:
                    for content in msg.get("content", []):
                        if content.get("type") == "tool_call":
                            any_tools_used = True
                            if "tool_call" in content and "function" in content["tool_call"]:
                                used_tool_names.add(content["tool_call"]["function"])
                            elif "name" in content:
                                used_tool_names.add(content["name"])

        filtered_tools = [tool for tool in tool_definitions if tool.get("name") in used_tool_names]
        if any_tools_used and not filtered_tools:
            if logger:
                logger.warning("No tool definitions matched the tools used in the messages. Returning original list.")
            filtered_tools = tool_definitions

        return filtered_tools
    except Exception as e:
        if logger:
            logger.warning(f"Failed to filter tool definitions, returning original list. Error: {e}")
        return tool_definitions


def _get_conversation_history(query, include_system_messages=False, include_tool_messages=False):
    """Parse conversation history from a list of messages into structured format.

    :param query: List of message dictionaries containing the conversation history
    :type query: List[dict]
    :param include_system_messages: Whether to include system messages in the output
    :type include_system_messages: bool
    :param include_tool_messages: Whether to include tool-related messages in agent responses
    :type include_tool_messages: bool
    :return: Dict containing parsed user_queries, agent_responses, and optionally system_message
    :rtype: Dict[str, Union[List[List[str]], str]]
    :raises EvaluationException: If conversation history is malformed (mismatched user/agent turns
    """
    all_user_queries, all_agent_responses = [], []
    cur_user_query, cur_agent_response = [], []
    system_message = None

    for msg in query:
        role = msg.get("role")
        if not role:
            continue
        if include_system_messages and role == "system":
            system_message = msg.get("content", "")

        elif role == "user" and "content" in msg:
            if cur_agent_response:
                formatted_agent_response = _get_agent_response(
                    cur_agent_response, include_tool_messages=include_tool_messages
                )
                all_agent_responses.append([formatted_agent_response])
                cur_agent_response = []
            text_in_msg = _extract_text_from_content(msg["content"])
            if text_in_msg:
                cur_user_query.append(text_in_msg)

        elif role in ("assistant", "tool"):
            if cur_user_query:
                all_user_queries.append(cur_user_query)
                cur_user_query = []
            cur_agent_response.append(msg)

    if cur_user_query:
        all_user_queries.append(cur_user_query)
    if cur_agent_response:
        formatted_agent_response = _get_agent_response(cur_agent_response, include_tool_messages=include_tool_messages)
        all_agent_responses.append([formatted_agent_response])

    if len(all_user_queries) != len(all_agent_responses) + 1:
        raise EvaluationException(
            message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            internal_message=ErrorMessage.MALFORMED_CONVERSATION_HISTORY,
            target=ErrorTarget.CONVERSATION_HISTORY_PARSING,
            category=ErrorCategory.INVALID_VALUE,
            blame=ErrorBlame.USER_ERROR,
        )

    result = {"user_queries": all_user_queries, "agent_responses": all_agent_responses}
    if include_system_messages and system_message:
        result["system_message"] = system_message
    return result


def _pretty_format_conversation_history(conversation_history):
    """Format the conversation history for better readability."""
    formatted_history = ""
    if conversation_history.get("system_message"):
        formatted_history += "SYSTEM_PROMPT:\n"
        formatted_history += "  " + conversation_history["system_message"] + "\n\n"
    for i, (user_query, agent_response) in enumerate(
        zip(
            conversation_history["user_queries"],
            conversation_history["agent_responses"] + [None],
        )
    ):
        formatted_history += f"User turn {i+1}:\n"
        for msg in user_query:
            formatted_history += "  " + "\n  ".join(msg)
        formatted_history += "\n\n"
        if agent_response:
            formatted_history += f"Agent turn {i+1}:\n"
            for msg in agent_response:
                formatted_history += "  " + "\n  ".join(msg)
            formatted_history += "\n\n"
    return formatted_history


def reformat_conversation_history(query, logger=None, include_system_messages=False, include_tool_messages=False):
    """Reformats the conversation history to a more compact representation."""
    try:
        conversation_history = _get_conversation_history(
            query,
            include_system_messages=include_system_messages,
            include_tool_messages=include_tool_messages,
        )
        return _pretty_format_conversation_history(conversation_history)
    except Exception as e:
        # If the conversation history cannot be parsed for whatever reason, the original query is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # However the accuracy of the evaluation will be affected.
        # From our tests the negative impact on IntentResolution is:
        #   Higher intra model variance (0.142 vs 0.046)
        #   Higher inter model variance (0.345 vs 0.607)
        #   Lower percentage of mode in Likert scale (73.4% vs 75.4%)
        #   Lower pairwise agreement between LLMs (85% vs 90% at the pass/fail level with threshold of 3)
        if logger:
            logger.warning(f"Conversation history could not be parsed, falling back to original query: {query}")
            print(e)
        return query


def _get_agent_response(agent_response_msgs, include_tool_messages=False):
    """Extract formatted agent response including text, and optionally tool calls/results."""
    agent_response_text = []
    tool_results = {}

    # First pass: collect tool results
    if include_tool_messages:
        for msg in agent_response_msgs:
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                for content in msg.get("content", []):
                    if content.get("type") == "tool_result":
                        result = content.get("tool_result")
                        tool_results[msg["tool_call_id"]] = f"[TOOL_RESULT] {result}"

    # Second pass: parse assistant messages and tool calls
    for msg in agent_response_msgs:
        if "role" in msg and msg.get("role") == "assistant" and "content" in msg:
            text = _extract_text_from_content(msg["content"])
            if text:
                agent_response_text.extend(text)
            if include_tool_messages:
                for content in msg.get("content", []):
                    # Todo: Verify if this is the correct way to handle tool calls
                    if content.get("type") == "tool_call":
                        if "tool_call" in content and "function" in content.get("tool_call", {}):
                            tc = content.get("tool_call", {})
                            func_name = tc.get("function", {}).get("name", "")
                            args = tc.get("function", {}).get("arguments", {})
                            tool_call_id = tc.get("id")
                        else:
                            tool_call_id = content.get("tool_call_id")
                            func_name = content.get("name", "")
                            args = content.get("arguments", {})
                        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
                        call_line = f"[TOOL_CALL] {func_name}({args_str})"
                        agent_response_text.append(call_line)
                        if tool_call_id in tool_results:
                            agent_response_text.append(tool_results[tool_call_id])

    return agent_response_text


def reformat_agent_response(response, logger=None, include_tool_messages=False):
    """Reformat agent response to a standardized string format.

    :param response: The agent response to reformat, can be None, empty list, or list of messages
    :type response: Union[None, List[dict], str]
    :param logger: Optional logger for warning messages
    :type logger: Optional[logging.Logger]
    :param include_tool_messages: Whether to include tool call and result information
    :type include_tool_messages: bool
    :return: Formatted agent response as a string, or original response if parsing fails
    :rtype: str
    """
    try:
        if response is None or response == []:
            return ""
        agent_response = _get_agent_response(response, include_tool_messages=include_tool_messages)
        if agent_response == []:
            # If no message could be extracted, fallback to the original response in that case
            if logger:
                logger.warning(
                    "Empty agent response extracted, likely due to input schema change. "
                    f"Falling back to using the original response: {response}"
                )
            return response
        return "\n".join(agent_response)
    except Exception as e:
        # If the agent response cannot be parsed for whatever reason (e.g. the converter format changed),
        # the original response is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(f"Agent response could not be parsed, falling back to original response. Error: {e}")
        return response


def reformat_tool_definitions(tool_definitions, logger=None):
    """Reformat tool definitions into a human-readable string format.

    :param tool_definitions: List of tool definition dictionaries containing name, description, and parameters
    :type tool_definitions: List[dict]
    :param logger: Optional logger for warning messages
    :type logger: Optional[logging.Logger]
    :return: Formatted tool definitions as a string, or original definitions if parsing fails
    :rtype: str
    """
    try:
        output_lines = ["TOOL_DEFINITIONS:"]
        for tool in tool_definitions:
            name = tool.get("name", "unnamed_tool")
            desc = tool.get("description", "").strip()
            params = tool.get("parameters", {}).get("properties", {})
            param_names = ", ".join(params.keys()) if params else "no parameters"
            output_lines.append(f"- {name}: {desc} (inputs: {param_names})")
        return "\n".join(output_lines)
    except Exception as e:
        # If the tool definitions cannot be parsed for whatever reason, the original tool definitions are returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(
                "Tool definitions could not be parsed, falling back to original definitions"
                f": {tool_definitions}. Error: {e}"
            )
        return tool_definitions


# ```


@experimental
class ToolOutputUtilizationEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluate how effectively an AI agent uses tool outputs.

    This evaluator checks whether the agent correctly incorporates information from tools into its responses.

    Scoring is based on two levels:
    1. Pass - effectively utilizes tool outputs and accurately incorporates the information into its response.
    2. Fail - fails to properly utilize tool outputs or incorrectly incorporates the information into its response.

    The evaluation includes the score, a brief explanation, and a final pass/fail result.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START tool_output_utilization_evaluator]
            :end-before: [END tool_output_utilization_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolOutputUtilizationEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START tool_output_utilization_evaluator]
            :end-before: [END tool_output_utilization_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call ToolOutputUtilizationEvaluator
                using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}
    """

    _PROMPTY_FILE = "tool_output_utilization.prompty"
    _RESULT_KEY = "tool_output_utilization"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _validator: ValidatorInterface

    _DEFAULT_TOOL_OUTPUT_UTILIZATION_SCORE = 1

    id = "azureai://built-in/evaluators/tool_output_utilization"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(
        self,
        model_config,
        *,
        credential=None,
        **kwargs,
    ):
        """Initialize the Tool Output Utilization Evaluator."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(
            error_target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            check_for_unsupported_tools=True,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=self._DEFAULT_TOOL_OUTPUT_UTILIZATION_SCORE,
            credential=credential,
            _higher_is_better=True,
            **kwargs,
        )

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        self._validator.validate_eval_input(kwargs)
        return await super()._real_call(**kwargs)

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
        tool_definitions: Union[dict, List[dict]],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate tool output utilization for a given query, response, and optional tool defintions.

        The query and response can be either a string or a list of messages.
        Example with string inputs and no tools:
            evaluator = ToolOutputUtilizationEvaluator(model_config)
            query = "What is the weather today?"
            response = "The weather is sunny."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = ToolOutputUtilizationEvaluator(model_config)
            query = [
                {
                    "role": "system",
                    "content": "You are a helpful customer service assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hi, can you check the status of my last order?",
                        }
                    ],
                },
            ]

            response = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sure! Let me look that up for you."}
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call": {
                                "id": "tool_1",
                                "type": "function",
                                "function": {
                                    "name": "get_order_status",
                                    "arguments": {"order_id": "123"},
                                },
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": '{"order_id": "123", "status": "shipped"}',
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "Your order 123 has been shipped and is on its way!",
                        }
                    ],
                },
            ]

            tool_definitions = [
                {
                    "name": "get_order_status",
                    "description": "Retrieve the status of an order by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {
                                "type": "string",
                                "description": "The order ID to check.",
                            }
                        },
                    },
                }
            ]


            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        (full agent response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :return: A dictionary with the tool output utilization evaluation results.
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

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Tool Output Utilization evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are needed for the _flow
            method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if ("query" not in eval_input) and ("response" not in eval_input) and ("tool_definitions" not in eval_input):
            raise EvaluationException(
                message=(
                    "Query, response, and tool_definitions are required inputs to "
                    "the Tool Output Utilization evaluator."
                ),
                internal_message=(
                    "Query, response, and tool_definitions are required inputs "
                    "to the Tool Output Utilization evaluator."
                ),
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
            )

        # If response or tool_definitions are strings, pass directly without reformatting
        # Process each parameter individually - strings pass through, dicts get reformatted
        tool_definitions = eval_input["tool_definitions"]
        if not isinstance(tool_definitions, str):
            if not isinstance(eval_input.get("query"), str) and not isinstance(eval_input.get("response"), str):
                filtered_tool_definitions = _filter_to_used_tools(
                    tool_definitions=tool_definitions,
                    msgs_lists=[eval_input["query"], eval_input["response"]],
                    logger=logger,
                )
            else:
                filtered_tool_definitions = tool_definitions
            eval_input["tool_definitions"] = reformat_tool_definitions(filtered_tool_definitions, logger)

        if not isinstance(eval_input.get("query"), str):
            eval_input["query"] = reformat_conversation_history(
                eval_input["query"],
                logger,
                include_system_messages=True,
                include_tool_messages=True,
            )
        if not isinstance(eval_input.get("response"), str):
            eval_input["response"] = reformat_agent_response(
                eval_input["response"], logger, include_tool_messages=True
            )

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
        if isinstance(llm_output, dict):
            output_label = llm_output.get("label", None)
            if output_label is None:
                if logger:
                    logger.warning("LLM output does not contain 'label' key, returning NaN for the score.")
                output_label = "fail"

            output_label = output_label.lower()
            if output_label not in ["pass", "fail"]:
                if logger:
                    logger.warning(
                        (
                            f"LLM output label is not 'pass' or 'fail' (got '{output_label}'), "
                            "returning NaN for the score."
                        )
                    )

            score = 1.0 if output_label == "pass" else 0.0
            score_result = output_label
            reason = llm_output.get("reason", "")

            faulty_details = llm_output.get("faulty_details", [])
            if faulty_details:
                reason += " Issues found: " + "; ".join(faulty_details)

            return {
                f"{self._result_key}": score,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }
        if logger:
            logger.warning("LLM output is not a dictionary, returning NaN for the score.")

        score = math.nan
        binary_result = self._get_binary_result(score)
        return {
            self._result_key: float(score),
            f"{self._result_key}_result": binary_result,
            f"{self._result_key}_threshold": self._threshold,
        }
