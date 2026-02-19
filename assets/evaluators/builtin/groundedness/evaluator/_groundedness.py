# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import math
from typing import Dict, List, Optional, Union, Any

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
)

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


class ConversationValidator(ValidatorInterface):
    """Validate conversation inputs (queries and responses) comprised of message lists."""

    requires_query: bool = True
    error_target: ErrorTarget

    def __init__(self, error_target: ErrorTarget, requires_query: bool = True):
        """Initialize with error target and query requirement."""
        self.requires_query = requires_query
        self.error_target = error_target

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
        if "type" not in content_item or content_item["type"] != ContentType.TOOL_CALL:
            return EvaluationException(
                message=f"The content item must be of type '{ContentType.TOOL_CALL.value}' in tool_call content item.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=self.error_target,
            )
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
            valid_assistant_content_types = [ContentType.TEXT, ContentType.OUTPUT_TEXT, ContentType.TOOL_CALL]
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
                else:
                    error = self._validate_tool_call_content_item(content_item)
                    if error:
                        return error
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
            if content_type != ContentType.TOOL_RESULT:
                return EvaluationException(
                    message=(
                        f"Invalid content type '{content_type}' for message with role '{MessageRole.TOOL.value}'. "
                        f"Must be '{ContentType.TOOL_RESULT.value}'."
                    ),
                    blame=ErrorBlame.USER_ERROR,
                    category=ErrorCategory.INVALID_VALUE,
                    target=self.error_target,
                )
            error = self._validate_dict_field(
                content_item, "tool_result", f"content items for role '{MessageRole.TOOL.value}'"
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
    _RESULT_KEY = "groundedness"
    _OPTIONAL_PARAMS = ["query"]
    _SUPPORTED_TOOLS = ["file_search"]

    _validator: ValidatorInterface
    _validator_with_query: ValidatorInterface

    id = "azureai://built-in/evaluators/groundedness"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=3, credential=None, **kwargs):
        """Initialize a GroundednessEvaluator instance."""
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE_NO_QUERY)  # Default to no query

        self._higher_is_better = True

        # Initialize input validator
        self._validator = ConversationValidator(error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR, requires_query=False)

        self._validator_with_query = ConversationValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR, requires_query=True
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
        # Needs to be set because it's used in call method to re-validate prompt if `query` is provided

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
        tool_definitions: List[dict],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate groundedness for agent response with tool calls. Only file_search tool is supported.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response from the agent to be evaluated.
        :paramtype response: List[dict]
        :keyword tool_definitions: The tool definitions used by the agent.
        :paramtype tool_definitions: List[dict]
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

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """Evaluate groundedness.

        Accepts either a query, response, and context for a single evaluation,
        or a conversation for a multi-turn evaluation.

        If the conversation has more than one turn, the evaluator will aggregate the results of each turn.

        :keyword query: The query to be evaluated. Mutually exclusive with `conversation`. Optional parameter for use
            with the `response` and `context` parameters. If provided, a different prompt template will be used for
            evaluation.
        :paramtype query: Optional[str]
        :keyword response: The response to be evaluated. Mutually exclusive with the `conversation` parameter.
        :paramtype response: Optional[str]
        :keyword context: The context to be evaluated. Mutually exclusive with the `conversation` parameter.
        :paramtype context: Optional[str]
        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The relevance score.
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

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
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

    async def _real_call(self, **kwargs):
        """Asynchronous call where real end-to-end evaluation logic is performed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Convert inputs into list of evaluable inputs.
        try:
            # Validate input before processing
            if kwargs.get("context") or kwargs.get("conversation"):
                self._validator.validate_eval_input(kwargs)
            else:
                self._validator_with_query.validate_eval_input(kwargs)
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
