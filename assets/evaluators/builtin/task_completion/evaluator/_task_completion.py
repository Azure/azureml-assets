# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from enum import Enum
from typing import Dict, Union, List, Optional, Any

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._common.constants import EvaluationLevel
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    validate_model_config,
    reformat_conversation_history,
    reformat_agent_response,
    reformat_tool_definitions,
    _resolve_evaluation_level,
    _is_intermediate_response,
    _preprocess_messages,
    _wrap_string_messages,
    _merge_query_response_messages,
    _split_messages_at_latest_user,
    serialize_messages,
)
from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    MessageRole,
    MessagesOrQueryResponseInputValidator,
)

if os.getenv("AI_EVALS_USE_PF_PROMPTY", "false").lower() == "true":
    from promptflow.core._flow import AsyncPrompty
else:
    from azure.ai.evaluation._legacy.prompty import AsyncPrompty


logger = logging.getLogger(__name__)


# Create extended ErrorTarget enum with the new member
def _create_extended_error_target():
    """Create an extended ErrorTarget enum that includes TASK_COMPLETION_EVALUATOR."""
    existing_members = {member.name: member.value for member in ErrorTarget}
    existing_members["TASK_COMPLETION_EVALUATOR"] = "TaskCompletionEvaluator"

    ExtendedErrorTarget = Enum("ExtendedErrorTarget", existing_members)
    return ExtendedErrorTarget


ExtendedErrorTarget = _create_extended_error_target()


@experimental
class TaskCompletionEvaluator(PromptyEvaluatorBase[Union[str, int]]):
    """The Task Completion evaluator determines whether an AI agent successfully completed the requested task.

    This evaluator assesses task completion based on:
        - Final outcome and deliverable of the task
        - Completeness of task requirements

    This evaluator focuses solely on task completion and success, not on task adherence or intent understanding.

    Scoring is binary:
    - 1 (Pass): Task fully completed with usable deliverable that meets all user requirements
    - 0 (Fail): Task incomplete, partially completed, or deliverable does not meet requirements

    The evaluation includes task requirement analysis, outcome assessment, and completion gap identification.


    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:
        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a TaskCompletionEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START task_completion_evaluator]
            :end-before: [END task_completion_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call TaskCompletionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "task_completion.prompty"
    _MULTI_TURN_PROMPTY_FILE = "task_completion_multi_turn.prompty"
    _RESULT_KEY = "task_completion"
    _OPTIONAL_PARAMS = ["messages", "tool_definitions"]

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/task_completion"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, credential=None, evaluation_level=None, **kwargs):
        """Initialize the TaskCompletionEvaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[AzureOpenAIModelConfiguration, OpenAIModelConfiguration]
        :keyword credential: Credential for authentication.
        :type credential: Optional[TokenCredential]
        :keyword evaluation_level: Force a specific evaluation level for this invocation. When ``None``
            (default), the level is auto-detected from input shape (``messages`` -> conversation,
            ``query``/``response`` -> turn). Set to ``EvaluationLevel.CONVERSATION`` or
            ``EvaluationLevel.TURN`` to override auto-detection.
        :type evaluation_level: Optional[Union[EvaluationLevel, str]]
        :keyword kwargs: Additional keyword arguments.
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", 1)

        # Validate and store evaluation level
        self._evaluation_level = _resolve_evaluation_level(
            evaluation_level, ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR
        )

        # Initialize input validator (supports both query/response and messages)
        self._validator = MessagesOrQueryResponseInputValidator(
            error_target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
        )

        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            credential=credential,
            threshold=threshold_value,
            **kwargs,
        )

        # Load the multi-turn prompty flow for multi-turn evaluation
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
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task completion for last agent response given a query, response, and optionally tool definitions.

        The query and response can be either a string or a list of messages.

        Example with string inputs and no tools:
            evaluator = TaskCompletionEvaluator(model_config)
            query = "Plan a 3-day itinerary for Paris with cultural landmarks and local cuisine."
            response = "**Day 1:** Morning: Louvre Museum, Lunch: Le Comptoir du Relais..."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = TaskCompletionEvaluator(model_config)
            query = [{'role': 'system', 'content': 'You are a helpful travel planning assistant.'},
                     {'createdAt': 1700000060, 'role': 'user',
                     'content': [{'type': 'text', 'text': 'Plan a 3-day Paris itinerary with cultural
                     landmarks and cuisine'}]}]
            response = [{'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
                        'content': [{'type': 'text', 'text': '*Day 1:* Morning: Visit Louvre Museum (9 AM-12 PM)'}]}]
            tool_definitions = [{'name': 'get_attractions', 'description': 'Get tourist attractions for a city.',
                                'parameters': {'type': 'object', 'properties':
                                  {'city': {'type': 'string', 'description': 'City name.'}}}}]

            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query being evaluated, either a string or a list of messages.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response being evaluated, either a string or a list of messages
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, int]]
        """

    @overload
    def __call__(
        self,
        *,
        messages: List[dict],
        tool_definitions: Optional[Union[dict, List[dict]]] = None,
    ) -> Dict[str, Union[str, int]]:
        """Evaluate task completion for a full multi-turn conversation.

        Example with messages and tool definitions:
            evaluator = TaskCompletionEvaluator(model_config)
            messages = [
                {"role": "system", "content": "You are a helpful travel assistant."},
                {"role": "user", "content": [{"type": "text", "text": "Find flights to London next Friday"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search for flights to London."},
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "name": "search_flights",
                            "arguments": {"origin": "NYC", "destination": "London", "date": "2025-01-10"},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_result": {"flights": [{"airline": "BA", "price": "$450", "departure": "8:00 AM"}]},
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "I found a British Airways flight at $450 departing 8:00 AM. Shall I book it?",
                        }
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Yes, book it."}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Done! Your flight is booked. Confirmation #BA12345."},
                    ],
                },
            ]
            tool_definitions = [
                {
                    "name": "search_flights",
                    "description": "Search for flights between two cities on a given date.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "origin": {"type": "string", "description": "Departure city"},
                            "destination": {"type": "string", "description": "Arrival city"},
                            "date": {"type": "string", "description": "Travel date (YYYY-MM-DD)"},
                        },
                    },
                }
            ]
            result = evaluator(messages=messages, tool_definitions=tool_definitions)

        :keyword messages: The full multi-turn conversation as a list of message dicts.
        :paramtype messages: List[dict]
        :keyword tool_definitions: An optional list of tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the task completion evaluation results.
        :rtype: Dict[str, Union[str, int]]
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

    def _build_result(
        self,
        score: Optional[int],
        result: str,
        reason: str,
        status: str,
        properties: Dict,
        prompty_output_dict: Optional[Dict] = None,
    ) -> Dict[str, Union[str, int, float, Dict, None]]:
        """Build a standardized result dictionary.

        :param score: The evaluation score (1, 0, or None).
        :param result: The result label ("pass", "fail", "skipped", or "error").
        :param reason: The reasoning or explanation string.
        :param status: The evaluation status ("completed", "skipped", or "error").
        :param properties: The properties dictionary.
        :param prompty_output_dict: Optional raw prompty output for extracting token metadata.
        :return: The standardized result dictionary.
        """
        p = prompty_output_dict if isinstance(prompty_output_dict, dict) else {}
        metadata = {
            "prompt_tokens": p.get("input_token_count", 0),
            "completion_tokens": p.get("output_token_count", 0),
            "total_tokens": p.get("total_token_count", 0),
            "finish_reason": p.get("finish_reason", ""),
            "model": p.get("model_id", ""),
            "sample_input": p.get("sample_input", ""),
            "sample_output": p.get("sample_output", ""),
        }
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_passed": result == "pass" if result in ["pass", "fail"] else None,
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_status": status,
            f"{self._result_key}_properties": {**properties, **metadata}
        }
        # Add top-level token metadata fields for backward compatibility.
        result_payload.update({f"{self._result_key}_{key}": value for key, value in metadata.items()})
        return result_payload

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
                kwargs.pop("messages", None)

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

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[int, str]]:  # type: ignore[override]
        """Do Task Completion evaluation.

        Routes to conversation-level or turn-level evaluation based on
        ``_evaluation_level`` (if set)
        or auto-detects from input shape (default).

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if self._should_use_conversation_level(eval_input):
            return await self._do_eval_conversation_level(eval_input)

        # Single-turn path (query/response)
        if eval_input.get("query") is None or eval_input.get("response") is None:
            raise EvaluationException(
                message="Both query and response must be provided as input to the Task Completion evaluator.",
                internal_message="Both query and response must be provided as input to the Task Completion evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
            )
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])
        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger, include_system_messages=True)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger, include_tool_messages=True)
        if "tool_definitions" in eval_input and eval_input["tool_definitions"]:
            eval_input["tool_definitions"] = reformat_tool_definitions(eval_input["tool_definitions"], logger)

        # Remove keys not consumed by the single-turn prompty to avoid leaking extra kwargs
        eval_input.pop("messages", None)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        return self._parse_prompty_output(prompty_output_dict)

    async def _do_eval_conversation_level(self, eval_input: Dict) -> Dict[str, Union[int, str]]:
        """Evaluate task completion for a full conversation-level evaluation.

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

    def _parse_prompty_output(self, prompty_output_dict: Dict) -> Dict[str, Union[int, str]]:
        """Parse the prompty output into a standardized result dictionary.

        Shared between single-turn and multi-turn evaluation paths.
        Expects the canonical schema: score (int), reason (str), status (str), properties (dict|null).

        :param prompty_output_dict: Raw output from the prompty flow.
        :type prompty_output_dict: Dict
        :return: The parsed evaluation result.
        :rtype: Dict
        """
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)

        if not isinstance(llm_output, dict):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ExtendedErrorTarget.TASK_COMPLETION_EVALUATOR,
            )

        # Handle skipped status from LLM
        llm_status = llm_output.get("status", "completed")
        if llm_status == "skipped":
            reason = llm_output.get("reason", "")
            return self._return_not_applicable_result(reason, self._threshold)

        score = float(llm_output.get("score", 0))
        success_result = "pass" if score >= 1.0 else "fail"
        reason = llm_output.get("reason", "")
        llm_properties = llm_output.get("properties", {}) or {}
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
