# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import math
import os
import re
from typing import Dict, List, Union
from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common._base_prompty_eval import PromptyEvaluatorBase
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._common.constants import PROMPT_BASED_REASON_EVALUATORS
from azure.ai.evaluation._common.utils import parse_quality_evaluator_reason_score

logger = logging.getLogger(__name__)


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


class RetrievalEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """
    Evaluate retrieval score for a given query and context or a multi-turn conversation, including reasoning.

    The retrieval measure assesses the AI system's performance in retrieving information
    for additional context (e.g. a RAG scenario).

    Retrieval scores range from 1 to 5, with 1 being the worst and 5 being the best.

    High retrieval scores indicate that the AI system has successfully extracted and ranked
    the most relevant information at the top, without introducing bias from external knowledge
    and ignoring factual correctness. Conversely, low retrieval scores suggest that the AI system
    has failed to surface the most relevant context chunks at the top of the list
    and/or introduced bias and ignored factual correctness.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the evaluation. Default is 3.
    :type threshold: float
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, the evaluator will use reasoning model configuration (o1/o3 models).
        This will adjust parameters like max_completion_tokens and remove unsupported parameters. Default is False.
    :paramtype is_reasoning_model: bool
    :return: A function that evaluates and generates metrics for "chat" scenario.
    :rtype: Callable

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START retrieval_evaluator]
            :end-before: [END retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a RetrievalEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START retrieval_evaluator]
            :end-before: [END retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call RetrievalEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_retrieval_evaluator]
            :end-before: [END threshold_retrieval_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a RetrievalEvaluator.

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    _PROMPTY_FILE = "retrieval.prompty"
    _RESULT_KEY = "retrieval"

    id = "azureai://built-in/evaluators/retrieval"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold: float = 3, credential=None, **kwargs):
        """Initialize the Retrieval evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: float
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(
            model_config=model_config,
            prompty_file=prompty_path,
            result_key=self._RESULT_KEY,
            threshold=threshold,
            credential=credential,
            _higher_is_better=self._higher_is_better,
            **kwargs,
        )

    @overload
    def __call__(
        self,
        *,
        query: str,
        context: str,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate retrieval for a given query and context.

        :keyword query: The query to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype query: Optional[str]
        :keyword context: The context to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype context: Optional[str]
        :return: The scores for Chat scenario.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate retrieval for a multi-turn evaluation.

        If the conversation has more than one turn,
        the evaluator will aggregate the results of each turn.

        :keyword conversation: The conversation to be evaluated.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The scores for Chat scenario.
        :rtype: Dict[str, Union[float, Dict[str, List[float]]]]
        """

    @override
    def __call__(self, *args, **kwargs):  # pylint: disable=docstring-missing-param
        """Evaluate retrieval score chat scenario.

        Accepts either a query and context for a single evaluation,
        or a conversation for a multi-turn evaluation. If the conversation has more than one turn,
        the evaluator will aggregate the results of each turn.

        :keyword query: The query to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype query: Optional[str]
        :keyword context: The context to be evaluated. Mutually exclusive with `conversation` parameter.
        :paramtype context: Optional[str]
        :keyword conversation: The conversation to be evaluated.
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The scores for Chat scenario.
        :rtype: :rtype: Dict[str, Union[float, Dict[str, List[str, float]]]]
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
                if str(llm_status).strip().lower() == "skipped":
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

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do a retrieval evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        if _is_intermediate_response(eval_input.get("response")):
            return self._return_not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if eval_input.get("response") is not None and isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        result = await self._the_super_do_eval(eval_input)
        # Check if base returned nan (invalid output case); None means not-applicable/skipped
        _score = result.get(self._result_key, 0)
        if _score is not None and math.isnan(_score):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.RETRIEVAL_EVALUATOR,
            )
        return result

    @override
    async def _real_call(self, **kwargs):
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
