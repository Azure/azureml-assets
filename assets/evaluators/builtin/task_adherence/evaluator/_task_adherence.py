# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import logging
from typing import Any, Dict, List, Optional, Union

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    construct_prompty_model_config,
    reformat_conversation_history,
    reformat_agent_response,
    reformat_tool_definitions,
    validate_model_config,
    _resolve_evaluation_level,
    _is_intermediate_response,
    _preprocess_messages,
    _wrap_string_messages,
    _merge_query_response_messages,
    _split_messages_at_latest_user,
    serialize_messages,
)
from azure.ai.evaluation._common.constants import EvaluationLevel
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
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
            error_target=ErrorTarget.TASK_ADHERENCE_EVALUATOR,
            deep_validate_messages=True,
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
        properties = dict(properties) if isinstance(properties, dict) else {}
        token_metadata = self._get_token_metadata(p)
        properties.update(token_metadata)
        result_payload = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_result": result,
            f"{self._result_key}_threshold": resolved_threshold,
            f"{self._result_key}_reason": reason,
            f"{self._result_key}_properties": properties,
        }
        # Add top-level token metadata fields for backward compatibility.
        result_payload.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
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
            return self._return_not_applicable_result(
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

        # Handle skipped status from LLM
        llm_status = llm_output.get("status", "completed")
        if llm_status == "skipped":
            reason = llm_output.get("reason", "")
            return self._return_not_applicable_result(reason, self._threshold)

        reasoning = llm_output.get("reason", "")
        score = float(llm_output.get("score", 0.0))
        score_result = "pass" if score >= 1.0 else "fail"
        llm_properties = llm_output.get("properties", {}) or {}
        token_metadata = self._get_token_metadata(prompty_output_dict)
        llm_properties.update(token_metadata)

        result = {
            self._result_key: score,
            f"{self._result_key}_score": score,
            f"{self._result_key}_passed": score_result == "pass",
            f"{self._result_key}_result": score_result,
            f"{self._result_key}_reason": reasoning,
            f"{self._result_key}_status": "completed",
            f"{self._result_key}_threshold": self._threshold,
            f"{self._result_key}_properties": llm_properties,
        }
        # Add top-level token metadata fields for backward compatibility.
        result.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
        return result
