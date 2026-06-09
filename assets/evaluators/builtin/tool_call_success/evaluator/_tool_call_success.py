# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
from typing import Dict, Union, List, Optional
from typing_extensions import overload, override
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from enum import Enum

from abc import ABC, abstractmethod
from typing import Any


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


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TOOL_CALL_SUCCESS_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TOOL_CALL_SUCCESS_EVALUATOR"] = "ToolCallSuccessEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


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
class ToolCallSuccessEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Tool Call Success evaluator determines whether tool calls done by an AI agent includes failures or not.

    This evaluator focuses solely on tool call results and tool definitions, disregarding user's query to
    the agent, conversation history and agent's final response. Although tool definitions is optional,
    providing them can help the evaluator better understand the context of the tool calls made by the
    agent. Please note that this evaluator validates tool calls for potential technical failures like
    errors, exceptions, timeouts and empty results (only in cases where empty results could indicate a
    failure). It does not assess the correctness or the tool result itself, like mathematical errors and
    unrealistic field values like name="668656".

    Scoring is binary:
    - TRUE: All tool calls were successful
    - FALSE: At least one tool call failed

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START TOOL_CALL_SUCCESS_EVALUATOR]
            :end-before: [END TOOL_CALL_SUCCESS_EVALUATOR]
            :language: python
            :dedent: 8
            :caption: Initialize and call a ToolCallSuccessEvaluator with a tool definitions and response.

    .. admonition:: Example using Azure AI Project URL:

    .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
        :start-after: [START TOOL_CALL_SUCCESS_EVALUATOR]
        :end-before: [END TOOL_CALL_SUCCESS_EVALUATOR]
        :language: python
        :dedent: 8
        :caption: Initialize and call ToolCallSuccessEvaluator using Azure AI Project URL in the following
            format https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "tool_call_success.prompty"
    _RESULT_KEY = "tool_call_success"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/tool_call_success"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize the Tool Call Success evaluator."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 1)
        higher_is_better_value = kwargs.pop("_higher_is_better", True)

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(
            error_target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=False,
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

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
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

    @overload
    def __call__(
        self,
        *,
        response: Union[str, List[dict]],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate tool call success for a given response, and optionally tool definitions.

        Example with list of messages:
            evaluator = ToolCallSuccessEvaluator(model_config)
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
            'content': [{'type': 'text', 'text': '**Day 1:** Morning: Visit Louvre Museum (9 AM - 12 PM)...'}]}]

            result = evaluator(response=response, )

        :keyword response: The response being evaluated, either a string or a list of messages (full agent
            response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: Optional tool definitions to use for evaluation.
        :paramtype tool_definitions: Union[dict, List[dict]]
        :return: A dictionary with the tool success evaluation results.
        :rtype: Dict[str, Union[str, float]]
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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[str, float]]:  # type: ignore[override]
        """Do Tool Call Success evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are
        needed for the _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "response" not in eval_input:
            raise EvaluationException(
                message="response, is a required inputs to the Tool Call Success evaluator.",
                internal_message="response, is a required inputs to the Tool Call Success evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if eval_input["response"] is None or eval_input["response"] == []:
            raise EvaluationException(
                message="response cannot be None or empty for the Tool Call Success evaluator.",
                internal_message="response cannot be None or empty for the Tool Call Success evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )

        # Short-circuit: if the agent runtime already reported a failed tool
        # execution via a known-failure ``status`` (e.g. "failed", "error",
        # "incomplete"), deterministically return ``fail`` without calling the
        # LLM. The evaluator's scoring contract is binary -- "FALSE: at least
        # one tool call failed" -- and the prompty rubric doesn't see the
        # ``status`` field, so it would otherwise grade only the (typically
        # empty) result body and frequently mis-score the conversation as a
        # pass. ``status`` is only populated by upstream converters that
        # preserve it; absent ``status``, behavior is unchanged.
        if isinstance(eval_input.get("response"), list):
            failed_statuses = _collect_failed_tool_statuses(eval_input["response"])
            if failed_statuses:
                reason = (
                    "Detected failed tool execution(s) with status "
                    + ", ".join(sorted(set(failed_statuses)))
                    + ". Marked as fail without LLM grading."
                )
                return {
                    self._result_key: 0.0,
                    f"{self._result_key}_score": 0.0,
                    f"{self._result_key}_passed": False,
                    f"{self._result_key}_result": "fail",
                    f"{self._result_key}_reason": reason,
                    f"{self._result_key}_status": "completed",
                    f"{self._result_key}_threshold": self._threshold,
                    f"{self._result_key}_properties": {
                        "short_circuit": "tool_status",
                        "failed_statuses": sorted(set(failed_statuses)),
                    },
                }

        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
            eval_input["tool_calls"] = _reformat_tool_calls_results(eval_input["response"], logger)
        # If response is a string, pass directly without reformatting
        elif isinstance(eval_input["response"], str):
            eval_input["tool_calls"] = eval_input["response"]
        else:
            raise EvaluationException(
                message="response must be either a list of messages or a string.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
            )

        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        # If tool definitions are string, pass directly without reformatting, else format it.
        if "tool_definitions" in eval_input and not isinstance(eval_input["tool_definitions"], str):
            tool_definitions = eval_input["tool_definitions"]
            # Only if response is not a string, we filter tool definitions to only tools needed.
            if not isinstance(eval_input["response"], str):
                tool_definitions = _filter_to_used_tools(
                    tool_definitions=tool_definitions,
                    msgs_list=eval_input["response"],
                    logger=logger,
                )
            eval_input["tool_definitions"] = _reformat_tool_definitions(tool_definitions, logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            # Handle skipped status from LLM
            llm_status = llm_output.get("status", "completed")
            if llm_status == "skipped":
                reason = llm_output.get("reason", "")
                return self._return_not_applicable_result(reason, self._threshold)

            llm_properties = llm_output.get("properties", {}) or {}

            score = float(llm_output.get("score", 0))
            success_result = "pass" if score >= 1.0 else "fail"
            reason = llm_output.get("reason", "")
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
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ExtendedErrorTarget.TOOL_CALL_SUCCESS_EVALUATOR,
        )


def _filter_to_used_tools(tool_definitions, msgs_list, logger=None):
    """Filter the tool definitions to only include those that were actually used in the messages lists."""
    try:
        used_tool_names = set()
        any_tools_used = False

        for msg in msgs_list:
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


def _format_value(v):
    if v is None:
        return "None"
    if isinstance(v, str):
        return f'"{v}"'
    return v


def _get_tool_calls_results(agent_response_msgs):
    """Extract formatted agent tool calls and results from response."""
    agent_response_text = []
    tool_results = {}

    # First pass: collect tool results

    for msg in agent_response_msgs:
        if msg.get("role") == "tool" and "tool_call_id" in msg:
            for content in msg.get("content", []):
                if content.get("type") == "tool_result":
                    result = content.get("tool_result")
                    tool_results[msg["tool_call_id"]] = f"[TOOL_RESULT] {result}"

    # Second pass: parse assistant messages and tool calls
    for msg in agent_response_msgs:
        if "role" in msg and msg.get("role") == "assistant" and "content" in msg:

            for content in msg.get("content", []):

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
                    args_str = ", ".join(f"{k}={_format_value(v)}" for k, v in args.items())
                    call_line = f"[TOOL_CALL] {func_name}({args_str})"
                    agent_response_text.append(call_line)
                    if tool_call_id in tool_results:
                        agent_response_text.append(tool_results[tool_call_id])

    return agent_response_text


def _reformat_tool_calls_results(response, logger=None):
    try:
        if response is None or response == []:
            return ""
        agent_response = _get_tool_calls_results(response)
        if agent_response == []:
            # If no message could be extracted, likely the format changed,
            # fallback to the original response in that case
            if logger:
                logger.warning(
                    f"Empty agent response extracted, likely due to input schema change. "
                    f"Falling back to using the original response: {response}"
                )
            return response
        return "\n".join(agent_response)
    except Exception as e:
        # If the agent response cannot be parsed for whatever
        # reason (e.g. the converter format changed), the original response is returned
        # This is a fallback to ensure that the evaluation can still proceed.
        # See comments on reformat_conversation_history for more details.
        if logger:
            logger.warning(f"Agent response could not be parsed, falling back to original response. Error: {e}")
            logger.debug(f"Original response: {response}")
        return response


def _reformat_tool_definitions(tool_definitions, logger=None):
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
                f"Tool definitions could not be parsed, falling back to original definitions. Error: {e}"
            )
            logger.debug(f"Original tool definitions: {tool_definitions}")
        return tool_definitions


_FAILED_TOOL_STATUSES = frozenset({"failed", "error", "incomplete", "cancelled", "canceled"})


def _collect_failed_tool_statuses(agent_response_msgs):
    """Return failure statuses seen on tool_call / tool_result content blocks.

    Scans ``agent_response_msgs`` for any ``tool_call`` or ``tool_result``
    content block whose ``status`` field is in ``_FAILED_TOOL_STATUSES``.

    Inputs are intentionally tolerated -- malformed messages / non-dict
    content blocks are skipped rather than raised on, so this helper is safe
    to call on freshly-deserialized agent traces.

    :param agent_response_msgs: The agent's response message list (already
        validated to be a list by the caller).
    :type agent_response_msgs: list
    :return: A list (with duplicates preserved) of lowercased failure status
        strings. Empty list means no failure signal was found.
    :rtype: list[str]
    """
    found = []
    if not isinstance(agent_response_msgs, list):
        return found
    for msg in agent_response_msgs:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in ("tool_call", "tool_result"):
                status = block.get("status")
                if isinstance(status, str) and status.lower() in _FAILED_TOOL_STATUSES:
                    found.append(status.lower())
    return found
