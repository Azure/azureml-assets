# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
from enum import Enum
from typing import Dict, Union, List

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import (
    EvaluationException,
    ErrorBlame,
    ErrorCategory,
    ErrorTarget,
)
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import (
    reformat_conversation_history,
    reformat_agent_response,
    reformat_tool_definitions,
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
            optional_tool_definitions=False,
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

        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

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
            # Handle skipped status from LLM
            llm_status = llm_output.get("status", "completed")
            if llm_status == "skipped":
                reason = llm_output.get("reason", "")
                return self._return_not_applicable_result(reason, self._threshold)

            score = float(llm_output.get("score", 0))
            score_result = "pass" if score >= 1.0 else "fail"
            reason = llm_output.get("reason", "")
            llm_properties = llm_output.get("properties", {}) or {}
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
            target=ErrorTarget.TOOL_OUTPUT_UTILIZATION_EVALUATOR,
        )
