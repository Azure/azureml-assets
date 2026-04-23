# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._model_configurations import Conversation

try:
    from azure.ai.evaluation._user_agent import UserAgentSingleton
except ImportError:

    class UserAgentSingleton:
        """Fallback singleton for user agent when import fails."""

        @property
        def value(self) -> str:
            """Return the user agent value."""
            return "None"


try:
    from azure.ai.evaluation._common.utils import construct_prompty_model_config, validate_model_config
except ImportError:
    from ..._common.utils import construct_prompty_model_config, validate_model_config

logger = logging.getLogger(__name__)


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
    error_target: ErrorTarget

    def __init__(
        self,
        error_target: ErrorTarget,
        requires_query: bool = True,
    ):
        """Initialize with error target and query requirement."""
        self.requires_query = requires_query
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


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes QUALITY_GRADER_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["QUALITY_GRADER_EVALUATOR"] = "QualityGraderEvaluator"

    _ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return _ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


# Thresholds for response quality checks (first prompt)
_QUALITY_RELEVANCE_THRESHOLD = 2.5
_ANSWER_COMPLETENESS_THRESHOLD = 1.5

# Thresholds for groundedness checks (second prompt)
_GROUNDEDNESS_THRESHOLD = 2.5
_CONTEXT_COVERAGE_THRESHOLD = 1.5


class QualityGraderEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluates overall response quality using a two-stage grading pipeline.

    Stage 1 (Response Quality): Evaluates the response for relevance, abstention, and answer completeness.
    The response must satisfy:
        - abstention must be false
        - relevance must be greater than 2.5 (on a 1-5 scale)
        - answerCompleteness must be greater than 1.5

    Stage 2 (Groundedness, only if context is provided): Evaluates whether the response is grounded in the
    provided context and covers the key information. The response must satisfy:
        - groundedness must be greater than 2.5
        - contextCoverage must exceed 1.5

    If all checks pass, the evaluator returns "pass". Otherwise, it returns "fail" with failure reasons.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, updates config parameters for reasoning models. Defaults to False.
    :paramtype is_reasoning_model: bool
    """

    _RESPONSE_QUALITY_PROMPTY = "quality_grader_response_quality.prompty"
    _GROUNDEDNESS_PROMPTY = "quality_grader_groundedness.prompty"
    _RESULT_KEY = "quality_grader"
    _OPTIONAL_PARAMS = ["context"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/quality_grader"

    @override
    def __init__(self, model_config, *, credential=None, **kwargs):
        """Initialize a QualityGraderEvaluator instance."""
        current_dir = os.path.dirname(__file__)
        response_quality_prompty_path = os.path.join(current_dir, self._RESPONSE_QUALITY_PROMPTY)

        self._higher_is_better = True
        self._model_config = model_config
        self._credential = credential

        # Initialize input validator
        self._validator = ConversationValidator(
            error_target=ExtendedErrorTarget.QUALITY_GRADER_EVALUATOR,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=response_quality_prompty_path,
            result_key=self._RESULT_KEY,
            threshold=1,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

        # Load the second prompty flow for groundedness evaluation
        groundedness_prompty_path = os.path.join(current_dir, self._GROUNDEDNESS_PROMPTY)
        subclass_name = self.__class__.__name__
        user_agent = f"{UserAgentSingleton().value} (type=evaluator subtype={subclass_name})"
        prompty_model_config = construct_prompty_model_config(
            validate_model_config(model_config),
            self._DEFAULT_OPEN_API_VERSION,
            user_agent,
        )
        self._groundedness_flow = AsyncPrompty.load(
            source=groundedness_prompty_path,
            model=prompty_model_config,
            token_credential=credential,
            is_reasoning_model=kwargs.get("is_reasoning_model", False),
        )

    @overload
    def __call__(
        self,
        *,
        query: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate quality for a given query, response, and optional context.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword context: The context (retrieved documents) to evaluate groundedness against. Optional.
        :paramtype context: Optional[str]
        :return: The quality grader result.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate quality for a conversation.

        :keyword conversation: The conversation to evaluate.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The quality grader result.
        :rtype: Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]
        """

    @override
    def __call__(self, *args, **kwargs):
        """Evaluate quality for a query/response pair with optional context for groundedness or a conversation.

        :return: The quality grader result.
        :rtype: Dict[str, Union[str, float]]
        """
        return super().__call__(*args, **kwargs)

    def _not_applicable_result(
        self, error_message: str,
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: 1.0,
            f"{self._result_key}_result": self._PASS_RESULT,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_details": {},
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_model": "",
        }

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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str, Dict]]:  # type: ignore[override]
        """Run the two-stage quality grading pipeline.

        Stage 1: Call the response quality prompt and check thresholds.
        Stage 2 (if context provided): Call the groundedness prompt and check thresholds.
        """
        # Handle intermediate responses
        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
            )

        # Preprocess messages
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        query = eval_input.get("query", "")
        response = eval_input.get("response", "")
        context = eval_input.get("context", None)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        model_id = ""

        # --- Stage 1: Response Quality ---
        stage1_input = {"question": query, "response": response}
        stage1_output = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **stage1_input)

        stage1_parsed = self._parse_prompty_json_output(stage1_output)
        total_prompt_tokens += stage1_output.get("input_token_count", 0) if stage1_output else 0
        total_completion_tokens += stage1_output.get("output_token_count", 0) if stage1_output else 0
        total_tokens += stage1_output.get("total_token_count", 0) if stage1_output else 0
        model_id = stage1_output.get("model_id", "") if stage1_output else ""

        # Check stage 1 conditions
        failure_reasons = []
        stage1_props = stage1_parsed.get("properties", {})
        abstention = stage1_props.get("abstention")
        relevance = stage1_props.get("relevance")
        answer_completeness = stage1_props.get("answerCompleteness")

        if abstention is True:
            failure_reasons.append("abstention is true (expected false)")
        if isinstance(relevance, (int, float)) and relevance <= _QUALITY_RELEVANCE_THRESHOLD:
            failure_reasons.append(
                f"relevance is {relevance} (must be > {_QUALITY_RELEVANCE_THRESHOLD})"
            )
        elif relevance is None or relevance == "null":
            failure_reasons.append(f"relevance is null (must be > {_QUALITY_RELEVANCE_THRESHOLD})")
        if isinstance(answer_completeness, (int, float)) and answer_completeness <= _ANSWER_COMPLETENESS_THRESHOLD:
            failure_reasons.append(
                f"answerCompleteness is {answer_completeness} (must be > {_ANSWER_COMPLETENESS_THRESHOLD})"
            )
        elif answer_completeness is None or answer_completeness == "null":
            failure_reasons.append(f"answerCompleteness is null (must be > {_ANSWER_COMPLETENESS_THRESHOLD})")

        if failure_reasons:
            return self._build_result(
                passed=False,
                failure_reasons=failure_reasons,
                stage1_parsed=stage1_parsed,
                stage2_parsed=None,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                model_id=model_id,
            )

        # --- Stage 2: Groundedness (only if context is provided) ---
        stage2_parsed = None
        if context:
            stage2_input = {"question": query, "response": response, "context": context}
            stage2_output = await self._groundedness_flow(timeout=self._LLM_CALL_TIMEOUT, **stage2_input)

            stage2_parsed = self._parse_prompty_json_output(stage2_output)
            total_prompt_tokens += stage2_output.get("input_token_count", 0) if stage2_output else 0
            total_completion_tokens += stage2_output.get("output_token_count", 0) if stage2_output else 0
            total_tokens += stage2_output.get("total_token_count", 0) if stage2_output else 0

            stage2_props = stage2_parsed.get("properties", {})
            groundedness = stage2_props.get("groundedness")
            context_coverage = stage2_props.get("contextCoverage")

            if isinstance(groundedness, (int, float)) and groundedness <= _GROUNDEDNESS_THRESHOLD:
                failure_reasons.append(
                    f"groundedness is {groundedness} (must be > {_GROUNDEDNESS_THRESHOLD})"
                )
            elif groundedness is None or groundedness == "null":
                failure_reasons.append(f"groundedness is null (must be > {_GROUNDEDNESS_THRESHOLD})")

            if isinstance(context_coverage, (int, float)) and context_coverage <= _CONTEXT_COVERAGE_THRESHOLD:
                failure_reasons.append(
                    f"contextCoverage is {context_coverage} (must exceed {_CONTEXT_COVERAGE_THRESHOLD})"
                )
            elif context_coverage is None or context_coverage == "null":
                failure_reasons.append(f"contextCoverage is null (must exceed {_CONTEXT_COVERAGE_THRESHOLD})")

            if failure_reasons:
                return self._build_result(
                    passed=False,
                    failure_reasons=failure_reasons,
                    stage1_parsed=stage1_parsed,
                    stage2_parsed=stage2_parsed,
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    total_tokens=total_tokens,
                    model_id=model_id,
                )

        # All checks passed
        return self._build_result(
            passed=True,
            failure_reasons=[],
            stage1_parsed=stage1_parsed,
            stage2_parsed=stage2_parsed,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            model_id=model_id,
        )

    @staticmethod
    def _parse_prompty_json_output(prompty_output: Optional[Dict]) -> Dict:
        """Parse the JSON output from a prompty flow call.

        :param prompty_output: The raw output dict from the prompty flow.
        :return: Parsed JSON dict from the LLM output.
        """
        if not prompty_output:
            return {}
        llm_output = prompty_output.get("llm_output", "")
        if not llm_output:
            return {}
        if isinstance(llm_output, dict):
            return llm_output
        try:
            return json.loads(llm_output)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse LLM output as JSON: %s", llm_output)
            return {}

    def _build_result(
        self,
        *,
        passed: bool,
        failure_reasons: List[str],
        stage1_parsed: Optional[Dict],
        stage2_parsed: Optional[Dict],
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        model_id: str,
    ) -> Dict[str, Union[str, float, Dict]]:
        """Build the standardized result dictionary.

        :param passed: Whether the evaluation passed.
        :param failure_reasons: List of reasons for failure (empty if passed).
        :param stage1_parsed: Parsed output from stage 1 (response quality).
        :param stage2_parsed: Parsed output from stage 2 (groundedness), or None if not run.
        :param prompt_tokens: Total prompt tokens used.
        :param completion_tokens: Total completion tokens used.
        :param total_tokens: Total tokens used.
        :param model_id: The model ID used.
        :return: Standardized result dict.
        """
        score = 1.0 if passed else 0.0
        result_label = self._PASS_RESULT if passed else self._FAIL_RESULT
        reason = "All quality checks passed." if passed else "; ".join(failure_reasons)

        details = {}
        if stage1_parsed:
            stage1_props = stage1_parsed.get("properties", {})
            details["abstention"] = stage1_props.get("abstention")
            details["relevance"] = stage1_props.get("relevance")
            details["answerCompleteness"] = stage1_props.get("answerCompleteness")
            details["queryType"] = stage1_props.get("queryType")
            details["conversationIncomplete"] = stage1_props.get("conversationIncomplete")
            details["judgeConfidence"] = stage1_props.get("judgeConfidence")
            details["stage1_explanation"] = stage1_props.get("explanation", {})
            details["stage1_reasoning"] = stage1_parsed.get("reasoning", "")
            details["stage1_score"] = stage1_parsed.get("score")
            details["stage1_status"] = stage1_parsed.get("status", "")

        if stage2_parsed:
            stage2_props = stage2_parsed.get("properties", {})
            details["groundedness"] = stage2_props.get("groundedness")
            details["contextCoverage"] = stage2_props.get("contextCoverage")
            details["documentUtility"] = stage2_props.get("documentUtility")
            details["missingContextParts"] = stage2_props.get("missingContextParts", [])
            details["unsupportedClaims"] = stage2_props.get("unsupportedClaims", [])
            details["stage2_explanation"] = stage2_props.get("explanation", {})
            details["stage2_reasoning"] = stage2_parsed.get("reasoning", "")
            details["stage2_score"] = stage2_parsed.get("score")
            details["stage2_status"] = stage2_parsed.get("status", "")

        return {
            self._result_key: score,
            f"{self._result_key}_result": result_label,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_details": details,
            f"{self._result_key}_prompt_tokens": prompt_tokens,
            f"{self._result_key}_completion_tokens": completion_tokens,
            f"{self._result_key}_total_tokens": total_tokens,
            f"{self._result_key}_model": model_id,
        }
