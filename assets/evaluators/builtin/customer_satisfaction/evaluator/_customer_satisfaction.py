# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import math
import os
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._common.utils import reformat_conversation_history, reformat_agent_response
from azure.ai.evaluation._common._experimental import experimental
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


# endregion Validators


# region Helper Functions


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


# endregion Helper Functions


logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes CUSTOMER_SATISFACTION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["CUSTOMER_SATISFACTION_EVALUATOR"] = "CustomerSatisfactionEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class CustomerSatisfactionEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """The Customer Satisfaction evaluator predicts customer satisfaction on a 1-5 Likert scale.

    This evaluator assesses whether an AI agent's response would likely result in a satisfied
    customer based on:
        - Helpfulness: Did the agent address what the user actually needed?
        - Completeness: Was the response thorough and complete?
        - Clarity: Was the response clear and easy to understand?
        - Tone: Was the tone appropriate, professional, and empathetic?
        - Resolution: Did the interaction resolve the user's issue?

    Scoring is on a 1-5 Likert scale:
    - 5: Very Satisfied - Issue fully resolved, excellent interaction
    - 4: Satisfied - Mostly addressed needs with minor gaps
    - 3: Neutral - Partially addressed needs, adequate response
    - 2: Dissatisfied - Failed to adequately address needs
    - 1: Very Dissatisfied - Completely failed to help or made things worse

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the evaluator. Default is 3.
    :type threshold: int

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START customer_satisfaction_evaluator]
            :end-before: [END customer_satisfaction_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a CustomerSatisfactionEvaluator with a query and response.

    """

    _PROMPTY_FILE = "customer_satisfaction.prompty"
    _RESULT_KEY = "customer_satisfaction"
    _OPTIONAL_PARAMS = []

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/customer_satisfaction"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, threshold=3, **kwargs):
        """Initialize the CustomerSatisfactionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword threshold: The threshold for the evaluator. Default is 3.
        :type threshold: int
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True

        # Initialize input validator
        self._validator = ConversationValidator(
            error_target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
            requires_query=True
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: Union[str, List[dict]],
        response: Union[str, List[dict]],
    ) -> Dict[str, Union[str, float]]:
        """Evaluate customer satisfaction for a given query and response.

        The query and response can be either a string or a list of messages.

        Example with string inputs:
            evaluator = CustomerSatisfactionEvaluator(model_config)
            query = "I need to cancel my order #12345"
            response = "I've cancelled your order. Refund will be processed in 3-5 days."

            result = evaluator(query=query, response=response)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :return: A dictionary with the customer satisfaction evaluation results.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate customer satisfaction for a conversation.

        :keyword conversation: The conversation to evaluate.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The customer satisfaction score
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
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

    def _not_applicable_result(
        self, error_message: str, threshold: Union[int, float]
    ) -> Dict[str, Union[str, float, Dict]]:
        """Return a result indicating that the evaluation is not applicable."""
        return {
            self._result_key: threshold,
            f"{self._result_key}_result": "pass",
            f"{self._result_key}_threshold": threshold,
            f"{self._result_key}_reason": f"Not applicable: {error_message}",
            f"{self._result_key}_dimensions": {},
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do Customer Satisfaction evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Customer Satisfaction evaluator.",
                internal_message="Both query and response required for Customer Satisfaction evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
            )

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )

        # Reformat inputs if they are lists of messages
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = reformat_conversation_history(
                eval_input["query"], logger, include_system_messages=True
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = reformat_agent_response(
                eval_input["response"], logger, include_tool_messages=True
            )

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if isinstance(llm_output, dict):
            score_value = llm_output.get("score", 3)
            if isinstance(score_value, str):
                score = float(score_value) if score_value.replace(".", "").isdigit() else 3.0
            else:
                score = float(score_value) if score_value else 3.0

            # Clamp score to 1-5 range
            score = max(1.0, min(5.0, score))

            success_result = "pass" if score >= self._threshold else "fail"
            reason = llm_output.get("explanation", "")
            dimensions = llm_output.get("dimensions", {})

            return {
                self._result_key: score,
                f"{self._result_key}_result": success_result,
                f"{self._result_key}_threshold": self._threshold,
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_dimensions": dimensions,
                f"{self._result_key}_prompt_tokens": prompty_output_dict.get("input_token_count", 0),
                f"{self._result_key}_completion_tokens": prompty_output_dict.get("output_token_count", 0),
                f"{self._result_key}_total_tokens": prompty_output_dict.get("total_token_count", 0),
                f"{self._result_key}_finish_reason": prompty_output_dict.get("finish_reason", ""),
                f"{self._result_key}_model": prompty_output_dict.get("model_id", ""),
                f"{self._result_key}_sample_input": prompty_output_dict.get("sample_input", ""),
                f"{self._result_key}_sample_output": prompty_output_dict.get("sample_output", ""),
            }

        # Check if base returned nan (invalid output case)
        if isinstance(llm_output, float) and math.isnan(llm_output):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
            )

        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ExtendedErrorTarget.CUSTOMER_SATISFACTION_EVALUATOR,
        )
