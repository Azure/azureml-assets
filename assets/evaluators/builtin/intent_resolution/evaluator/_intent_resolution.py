# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import math
import logging
from typing import Dict, Union, List, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    check_score_is_valid,
    reformat_conversation_history,
    reformat_agent_response,
    _is_intermediate_response,
    _preprocess_messages,
)
from azure.ai.evaluation._common._experimental import experimental
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

from azure.ai.evaluation._evaluators._common._validators import (
    ValidatorInterface,
    ToolDefinitionsValidator,
)


logger = logging.getLogger(__name__)


@experimental
class IntentResolutionEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """
    Evaluates intent resolution for a given query and response or a multi-turn conversation, including reasoning.

    The intent resolution evaluator assesses whether the user intent was correctly identified and resolved.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START intent_resolution_evaluator]
            :end-before: [END intent_resolution_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call an IntentResolutionEvaluator with a query and response.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START intent_resolution_evaluator]
            :end-before: [END intent_resolution_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call IntentResolutionEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    _PROMPTY_FILE = "intent_resolution.prompty"
    _RESULT_KEY = "intent_resolution"
    _OPTIONAL_PARAMS = ["tool_definitions"]

    _MIN_INTENT_RESOLUTION_SCORE = 1
    _MAX_INTENT_RESOLUTION_SCORE = 5
    _DEFAULT_INTENT_RESOLUTION_THRESHOLD = 3

    _validator: ValidatorInterface

    id = "azureai://built-in/evaluators/intent_resolution"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=_DEFAULT_INTENT_RESOLUTION_THRESHOLD, credential=None, **kwargs):
        """Initialize the Intent Resolution evaluator.

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
        higher_is_better_value = kwargs.pop("_higher_is_better", True)
        self.threshold = threshold_value

        # Initialize input validator
        self._validator = ToolDefinitionsValidator(error_target=ErrorTarget.INTENT_RESOLUTION_EVALUATOR)

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
        """Evaluate intent resolution for a given query, response and optional tool definitions.

        The query and response can be either a string or a list of messages.

        Example with string inputs and no tools:
            evaluator = IntentResolutionEvaluator(model_config)
            query = "What is the weather today?"
            response = "The weather is sunny."

            result = evaluator(query=query, response=response)

        Example with list of messages:
            evaluator = IntentResolutionEvaluator(model_config)
            query: [
                {'role': 'system', 'content': 'You are a friendly and helpful customer service agent.'},
                {'createdAt': 1700000060, 'role': 'user', 'content': [{'type': 'text',
                 'text': 'Hi, I need help with the last 2 orders on my account #888. Could you please update me '
                         'on their status?'}]}
            ]
            response: [
                {'createdAt': 1700000070, 'run_id': '0', 'role': 'assistant',
                 'content': [{'type': 'text', 'text': 'Hello! Let me quickly look up your account details.'}]},
                {'createdAt': 1700000075, 'run_id': '0', 'role': 'assistant',
                 'content': [{'type': 'tool_call',
                              'tool_call': {'id': 'tool_call_20250310_001', 'type': 'function',
                              'function': {'name': 'get_orders', 'arguments': {'account_number': '888'}}}}]},
                {'createdAt': 1700000080, 'run_id': '0', 'tool_call_id': 'tool_call_20250310_001', 'role': 'tool',
                 'content': [{'type': 'tool_result',
                              'tool_result': '[{ "order_id": "123" }, { "order_id": "124" }]'}]},
                {'createdAt': 1700000085, 'run_id': '0', 'role': 'assistant',
                 'content': [{'type': 'text',
                              'text': 'Thanks for your patience. I see two orders on your account. '
                                      'Let me fetch the details for both.'}]},
                {'createdAt': 1700000090, 'run_id': '0', 'role': 'assistant', 'content': [
                    {'type': 'tool_call',
                     'tool_call': {'id': 'tool_call_20250310_002', 'type': 'function',
                                   'function': {'name': 'get_order', 'arguments': {'order_id': '123'}}}},
                    {'type': 'tool_call',
                     'tool_call': {'id': 'tool_call_20250310_003', 'type': 'function',
                                   'function': {'name': 'get_order',
                                               'arguments': {'order_id': '124'}}}}
                ]},
                {'createdAt': 1700000095, 'run_id': '0', 'tool_call_id': 'tool_call_20250310_002', 'role': 'tool',
                 'content': [{'type': 'tool_result',
                              'tool_result': '{ "order": { "id": "123", "status": "shipped", '
                                             '"delivery_date": "2025-03-15" } }'}]},
                {'createdAt': 1700000100, 'run_id': '0', 'tool_call_id': 'tool_call_20250310_003', 'role': 'tool',
                 'content': [{'type': 'tool_result',
                              'tool_result': '{ "order": { "id": "124", "status": "delayed", '
                                             '"expected_delivery": "2025-03-20" } }'}]},
                {'createdAt': 1700000105, 'run_id': '0', 'role': 'assistant', 'content': [{'type': 'text',
                 'text': 'The order with ID 123 has been shipped and is expected to be delivered on March 15, 2025. '
                         'However, the order with ID 124 is delayed and should now arrive by March 20, 2025. '
                         'Is there anything else I can help you with?'}]}
            ]
            tool_definitions: [
                {'name': 'get_orders', 'description': 'Get the list of orders for a given account number.',
                 'parameters': {'type': 'object', 'properties': {'account_number': {'type': 'string',
                 'description': 'The account number to get the orders for.'}}}},
                {'name': 'get_order', 'description': 'Get the details of a specific order.',
                 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string',
                 'description': 'The order ID to get the details for.'}}}},
                {'name': 'initiate_return', 'description': 'Initiate the return process for an order.',
                 'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string',
                 'description': 'The order ID for the return process.'}}}},
                {'name': 'update_shipping_address', 'description': 'Update the shipping address for a given account.',
                 'parameters': {'type': 'object', 'properties': {'account_number': {'type': 'string',
                 'description': 'The account number to update.'}, 'new_address': {'type': 'string',
                 'description': 'The new shipping address.'}}}}
            ]

            result = evaluator(query=query, response=response, tool_definitions=tool_definitions)

        :keyword query: The query to be evaluated which is either a string or a list of messages.
            The list of messages is the previous conversation history of the user and agent, including system
            messages and tool calls.
        :paramtype query: Union[str, List[dict]]
        :keyword response: The response to be evaluated, which is either a string or a list of messages
            (full agent response potentially including tool calls)
        :paramtype response: Union[str, List[dict]]
        :keyword tool_definitions: An optional list of messages containing the tool definitions the agent is aware of.
        :paramtype tool_definitions: Optional[Union[dict, List[dict]]]
        :return: A dictionary with the intent resolution evaluation
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
    async def _real_call(self, **kwargs):
        """Perform asynchronous call where real end-to-end evaluation logic is executed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        # Validate input before processing
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
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do intent resolution evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are needed for the
            _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # we override the _do_eval method as we want the output to be a dictionary, which is a different schema
        # than _base_prompty_eval.py
        if "query" not in eval_input and "response" not in eval_input:
            raise EvaluationException(
                message="Both query and response must be provided as input to the intent resolution evaluator.",
                internal_message="Both query and response must be provided as input to the intent resolution "
                "evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.INTENT_RESOLUTION_EVALUATOR,
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
        # reformat query and response to the format expected by the prompty flow
        eval_input["query"] = reformat_conversation_history(eval_input["query"], logger)
        eval_input["response"] = reformat_agent_response(eval_input["response"], logger)

        prompty_output_dict = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = prompty_output_dict.get("llm_output", prompty_output_dict)
        score = math.nan
        if isinstance(llm_output, dict):
            # Handle skipped status from LLM
            llm_status = llm_output.get("status", "completed")
            if llm_status == "skipped":
                reason = llm_output.get("reason", "")
                return self._return_not_applicable_result(reason, self._threshold)

            score = llm_output.get("score", math.nan)
            if not check_score_is_valid(
                score,
                IntentResolutionEvaluator._MIN_INTENT_RESOLUTION_SCORE,
                IntentResolutionEvaluator._MAX_INTENT_RESOLUTION_SCORE,
            ):
                raise EvaluationException(
                    message=f"Invalid score value: {score}. Expected a number in range "
                    f"[{IntentResolutionEvaluator._MIN_INTENT_RESOLUTION_SCORE}, "
                    f"{IntentResolutionEvaluator._MAX_INTENT_RESOLUTION_SCORE}].",
                    internal_message="Invalid score value.",
                    category=ErrorCategory.FAILED_EXECUTION,
                    blame=ErrorBlame.SYSTEM_ERROR,
                )
            reason = llm_output.get("reason", "")
            score = float(score)
            score_result = "pass" if score >= self._threshold else "fail"
            llm_properties = llm_output.get("properties", {}) or {}
            token_metadata = self._get_token_metadata(prompty_output_dict)
            llm_properties.update(token_metadata)

            response_dict = {
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
            response_dict.update({f"{self._result_key}_{key}": value for key, value in token_metadata.items()})
            return response_dict
        raise EvaluationException(
            message="Evaluator returned invalid output.",
            blame=ErrorBlame.SYSTEM_ERROR,
            category=ErrorCategory.FAILED_EXECUTION,
            target=ErrorTarget.INTENT_RESOLUTION_EVALUATOR,
        )
