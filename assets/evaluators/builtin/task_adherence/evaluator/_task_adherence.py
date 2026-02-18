# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    reformat_conversation_history,
    reformat_agent_response,
)
from azure.ai.evaluation._common._experimental import experimental

from abc import ABC, abstractmethod
from enum import Enum
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


# endregion Validators

logger = logging.getLogger(__name__)


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
    _RESULT_KEY = "task_adherence"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _DEFAULT_TASK_ADHERENCE_SCORE = 0

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/task_adherence"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=_DEFAULT_TASK_ADHERENCE_SCORE, credential=None, **kwargs):
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

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(error_target=ErrorTarget.TASK_ADHERENCE_EVALUATOR)

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold_value,
            credential=credential,
            _higher_is_better=higher_is_better_value,
            **kwargs,
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

    @override
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
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
        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if "query" not in eval_input or "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Task Adherence evaluator.",
                internal_message="Both query and response must be provided as input to the Task Adherence evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.TASK_ADHERENCE_EVALUATOR,
            )

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

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **prompty_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            flagged = llm_output.get("flagged", False)
            reasoning = llm_output.get("reasoning", "")
            # Convert flagged to numeric score for backward compatibility (1 = pass, 0 = fail)
            score = 0.0 if flagged else 1.0
            score_result = "fail" if flagged else "pass"

            return {
                f"{self._result_key}": score,
                f"{self._result_key}_result": score_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reasoning,
                f"{self._result_key}_details": llm_output.get("details", ""),
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }

        if logger:
            logger.warning("LLM output is not a dictionary, returning 0 for the success.")

        return {self._result_key: 0}
