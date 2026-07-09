# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import logging
import math
import re
from typing import Dict, List, Optional, Union, Any

from typing_extensions import overload, override

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._common.constants import PROMPT_BASED_REASON_EVALUATORS, EvaluationLevel
from azure.ai.evaluation._common.utils import (
    ErrorBlame,
    ErrorTarget,
    EvaluationException,
    ErrorCategory,
    check_score_is_valid,
    construct_prompty_model_config,
    validate_model_config,
    parse_quality_evaluator_reason_score,
    _extract_text_from_content,
    _resolve_evaluation_level,
    _is_intermediate_response,
    _preprocess_messages,
    _wrap_string_messages,
    _merge_query_response_messages,
    _split_messages_at_latest_user,
    serialize_messages,
)
from azure.ai.evaluation._common.utils import reformat_tool_definitions
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    MessageRole,
    ConversationValidator,
    MessagesOrQueryResponseInputValidator,
)

try:
    from azure.ai.evaluation._evaluators._common._validators import (
        GroundednessConversationValidator,
    )
except ImportError:
    class GroundednessConversationValidator(ConversationValidator):
        """Fallback used when the installed SDK lacks GroundednessConversationValidator.

        Groundedness keeps rejecting ``azure_ai_search`` / ``azure_fabric`` /
        ``sharepoint_grounding`` tool calls, whose structured results the
        groundedness judge cannot yet consume as context.
        """

        UNSUPPORTED_TOOLS: List[str] = ConversationValidator.UNSUPPORTED_TOOLS + [
            "azure_ai_search",
            "azure_fabric",
            "sharepoint_grounding",
        ]


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
    _MULTI_TURN_PROMPTY_FILE = "groundedness_multi_turn.prompty"
    _RESULT_KEY = "groundedness"
    _MIN_GROUNDEDNESS_SCORE = 1
    _MAX_GROUNDEDNESS_SCORE = 5
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
            ``query``/``response`` -> turn). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TURN`` to override auto-detection.
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
        self._validator = GroundednessConversationValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=True
        )

        self._validator_with_query = GroundednessConversationValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR, requires_query=True,
            check_for_unsupported_tools=True
        )

        self._validator_messages = MessagesOrQueryResponseInputValidator(
            error_target=ErrorTarget.GROUNDEDNESS_EVALUATOR,
            requires_query=False,
            check_for_unsupported_tools=False,
            enforce_tool_definitions=False,
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

    def _build_result(
        self,
        score: Optional[Union[int, float]],
        result: str,
        reason: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized groundedness result dictionary."""
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        properties = dict(properties) if isinstance(properties, dict) else {}
        token_metadata = self._get_token_metadata(p)
        properties.update(token_metadata)
        parsed_result: Dict[str, Union[str, int, float, Dict, None]] = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_threshold": self.threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_properties": properties,
        }
        if status is not None:
            parsed_result[f"{self._result_key}_status"] = status
        # Add top-level token metadata fields for backward compatibility.
        parsed_result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return parsed_result

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

    async def _the_super_do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        """Do a relevance evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Only text conversation inputs are supported.",
                internal_message="Only text conversation inputs are supported.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.INVALID_VALUE,
                target=ErrorTarget.CONVERSATION,
            )
        # Check for intermediate response
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        # Preprocess messages if they are lists
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        # Call the prompty flow to get the evaluation result.
        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        score = math.nan
        reason = ""
        llm_properties = {}
        if prompty_output_dict:
            llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
            parsed_output = None
            if isinstance(llm_output, dict):
                parsed_output = llm_output
            elif isinstance(llm_output, str):
                try:
                    parsed_output = json.loads(llm_output)
                except (json.JSONDecodeError, TypeError):
                    parsed_output = None
            if parsed_output and isinstance(parsed_output, dict):
                llm_status = parsed_output.get("status", "completed")
                if llm_status == "skipped":
                    skip_reason = parsed_output.get("reason", "")
                    return self._return_not_applicable_result(skip_reason, self._threshold)
                score = parsed_output.get("score", math.nan)
                reason = parsed_output.get("reason", "")
                llm_properties = parsed_output.get("properties", {}) or {}
            else:
                if isinstance(llm_output, str) and self._result_key in PROMPT_BASED_REASON_EVALUATORS:
                    score, reason = parse_quality_evaluator_reason_score(llm_output)
                elif isinstance(llm_output, str):
                    match = re.search(r"\d", llm_output)
                    if match:
                        score = float(match.group())
            score = float(score) if score is not None else math.nan
            score_result = self._get_binary_result(score)
            token_metadata = self._get_token_metadata(prompty_output_dict)
            llm_properties.update(token_metadata)
            result = {
                self._result_key: score,
                f"{self._result_key}_score": score,
                f"{self._result_key}_passed": score_result == "pass",
                f"{self._result_key}_result": score_result,
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
            target=ErrorTarget.EVALUATE,
        )

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
        if self._evaluation_level == EvaluationLevel.TURN:
            return False
        # Auto-detect (_evaluation_level is None)
        return eval_input.get("messages") is not None

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:
        # Route to conversation-level evaluation if appropriate
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self.threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        if eval_input.get("query", None) is None:
            result = await self._the_super_do_eval(eval_input)
            # Check if base returned nan (invalid output case); None means not-applicable/skipped
            _score = result.get(self._result_key, 0)
            if _score is not None and math.isnan(_score):
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
        result = await self._the_super_do_eval(simplified_eval_input)
        # Check if base returned nan (invalid output case); None means not-applicable/skipped
        _score = result.get(self._result_key, 0)
        if _score is not None and math.isnan(_score):
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
        return self._parse_prompty_output(prompty_output_dict)

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Any]:
        """Parse the prompty output into a standardized result dictionary.

        :param prompty_output_dict: Raw output from the prompty flow.
        :type prompty_output_dict: Dict
        :return: The parsed evaluation result.
        :rtype: Dict
        """
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        score: Optional[Union[int, float]] = None
        score_result = "error"
        reason = "Evaluator returned invalid output."
        status = "error"
        properties: Dict[str, Any] = {}

        if isinstance(llm_output, dict):
            status = str(llm_output.get("status", "completed")).strip().lower()
            reason = llm_output.get("reason", llm_output.get("explanation", ""))
            properties = llm_output.get("properties", llm_output.get("properties", {})) or {}
            if not isinstance(properties, dict):
                properties = {}

            if status in ["skipped", "error"]:
                score = None
                score_result = "not_applicable"
            else:
                score_value = llm_output.get("score", self.threshold)
                if isinstance(score_value, str):
                    normalized_score = score_value.strip()
                    score = float(normalized_score) if normalized_score.replace(".", "", 1).isdigit() else None
                elif isinstance(score_value, (int, float)):
                    score = float(score_value)
                else:
                    score = None

                if score is None or not check_score_is_valid(
                    score,
                    GroundednessEvaluator._MIN_GROUNDEDNESS_SCORE,
                    GroundednessEvaluator._MAX_GROUNDEDNESS_SCORE,
                ):
                    score_result = "error"
                    reason = reason or (
                        f"Invalid score value: {score}. Expected a number in range "
                        f"[{GroundednessEvaluator._MIN_GROUNDEDNESS_SCORE}, "
                        f"{GroundednessEvaluator._MAX_GROUNDEDNESS_SCORE}]."
                    )
                    status = "error"
                else:
                    score_result = "pass" if score >= self.threshold else "fail"

        return self._build_result(
            score=score,
            result=score_result,
            reason=reason,
            properties=properties,
            status=status,
            prompty_output_dict=prompty_output_dict,
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
        elif self._evaluation_level == EvaluationLevel.TURN and kwargs.get("messages"):
            if any(m.get("role") == MessageRole.USER for m in kwargs["messages"]):
                query_messages, response_messages = _split_messages_at_latest_user(kwargs["messages"])
                kwargs["query"] = query_messages
                kwargs["response"] = response_messages
                kwargs.pop("messages", None)

        # Validate input before processing
        if kwargs.get("messages"):
            self._validator_messages.validate_eval_input(kwargs)
        elif kwargs.get("query"):
            self._validator_with_query.validate_eval_input(kwargs)
        else:
            self._validator.validate_eval_input(kwargs)

        # Convert inputs into list of evaluable inputs.
        try:
            return await self._the_super_real_call(**kwargs)
        except EvaluationException as ex:
            if ex.category == ErrorCategory.NOT_APPLICABLE:
                return self._return_not_applicable_result(ex.message, self.threshold)
            else:
                raise ex

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
                                    blame=ErrorBlame.USER_ERROR,
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
            logger.debug("Tool calls parsed successfully: count=%d", len(tool_calls) if tool_calls else 0)

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
