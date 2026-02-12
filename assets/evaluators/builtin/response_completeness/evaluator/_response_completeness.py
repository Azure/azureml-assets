# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import math
from typing import Dict, List, Union, Optional

from typing_extensions import overload, override

from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._common.utils import parse_quality_evaluator_reason_score
from azure.ai.evaluation._model_configurations import Conversation
from azure.ai.evaluation._common._experimental import experimental

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
    """Normalize function_call/function_call_output types to tool_call/tool_result."""
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
    return messages


def _preprocess_messages(messages):
    """Drop MCP approval messages and normalize function call types."""
    messages = _drop_mcp_approval_messages(messages)
    messages = _normalize_function_call_types(messages)
    return messages


@experimental
class ResponseCompletenessEvaluator(PromptyEvaluatorBase[Union[str, float]]):
    """Evaluate the extent to which a response contains necessary and relevant information with respect to truth.

    The completeness measure assesses how thoroughly an AI model's generated response aligns with the key information,
    claims, and statements established in the ground truth. This evaluation considers the presence, accuracy,
    and relevance of the content provided.
    The assessment spans multiple levels, ranging from fully incomplete to fully complete, ensuring a comprehensive
    evaluation of the response's content quality.
    Use this metric when you need to evaluate an AI model's ability to deliver comprehensive and accurate information,
    particularly in text generation tasks where conveying all essential details is crucial for clarity,
    context, and correctness.
    Completeness scores range from 1 to 5:
    1: Fully incomplete — Contains none of the necessary information.
    2: Barely complete — Contains only a small portion of the required information.
    3: Moderately complete — Covers about half of the required content.
    4: Mostly complete — Includes most of the necessary details with minimal omissions.
    5: Fully complete — Contains all key information without any omissions.
    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START completeness_evaluator]
            :end-before: [END completeness_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a CompletenessEvaluator with a response and groundtruth.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START completeness_evaluator]
            :end-before: [END completeness_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call CompletenessEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    """

    # Constants must be defined within eval's directory to be save/loadable

    _PROMPTY_FILE = "response_completeness.prompty"
    _RESULT_KEY = "response_completeness"

    id = "azureai://built-in/evaluators/response_completeness"

    _MIN_COMPLETENESS_SCORE = 1
    _MAX_COMPLETENESS_SCORE = 5
    _DEFAULT_COMPLETENESS_THRESHOLD = 3

    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(
        self, model_config, *, threshold: Optional[float] = _DEFAULT_COMPLETENESS_THRESHOLD, credential=None, **kwargs
    ):
        """Initialize the Response Completeness evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: Optional[float]
        :param credential: The credential for authentication.
        :type credential: Optional[Any]
        :param kwargs: Additional keyword arguments.
        :type kwargs: Any
        """
        current_dir = os.path.dirname(__file__)
        prompty_path = os.path.join(current_dir, self._PROMPTY_FILE)
        threshold_value = kwargs.pop("threshold", threshold)
        higher_is_better_value = kwargs.pop("_higher_is_better", True)
        self.threshold = threshold_value  # to be removed in favor of _threshold
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
        ground_truth: str,
        response: str,
    ) -> Dict[str, Union[str, float]]:
        """Evaluate completeness in given response.

        Accepts ground truth and response for evaluation.
        Example usage:
        Evaluating completeness for a response string
        ```python
        from azure.ai.evaluation import CompletenessEvaluator
        completeness_evaluator = CompletenessEvaluator(model_config)
        ground_truth = "The ground truth to be evaluated."
        response = "The response to be evaluated."
        completeness_results = completeness_evaluator(ground_truth=ground_truth, response=response)
        ```
        :keword ground_truth: The ground truth to be evaluated.
        :paramtype ground_truth: str
        :keyword response: The response to be evaluated.
        :paramtype response: Union[str, List[Message]]
        :return: The response completeness score results.
        :rtype: Dict[str, Union[str, float]]
        """

    @overload
    def __call__(
        self,
        *,
        conversation: Conversation,
    ) -> Dict[str, Union[float, Dict[str, List[Union[str, float]]]]]:
        """Evaluate completeness for a conversation.

        :keyword conversation: The conversation to evaluate. Expected to contain a list of conversation turns under the
            key "messages", and potentially a global context under the key "context". Conversation turns are expected
            to be dictionaries with keys "content", "role", and possibly "context".
        :paramtype conversation: Optional[~azure.ai.evaluation.Conversation]
        :return: The fluency score
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
            f"{self._result_key}_prompt_tokens": 0,
            f"{self._result_key}_completion_tokens": 0,
            f"{self._result_key}_total_tokens": 0,
            f"{self._result_key}_finish_reason": "",
            f"{self._result_key}_model": "",
            f"{self._result_key}_sample_input": "",
            f"{self._result_key}_sample_output": "",
        }

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, Union[float, str]]:  # type: ignore[override]
        """Do completeness evaluation.

        :param eval_input: The input to the evaluator. Expected to contain whatever inputs are needed for the
            _flow method
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        # we override the _do_eval method as we want the output to be a dictionary,
        # which is a different schema than _base_prompty_eval.py
        if "ground_truth" not in eval_input or "response" not in eval_input:
            raise EvaluationException(
                message="Both ground_truth and response must be provided as input to the completeness evaluator.",
                internal_message="Both ground_truth and response must be provided as input to the completeness "
                "evaluator.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.COMPLETENESS_EVALUATOR,
            )

        if _is_intermediate_response(eval_input.get("response")):
            return self._not_applicable_result(
                "Intermediate response. Please provide the agent's final response for evaluation.",
                self._threshold,
            )
        if isinstance(eval_input.get("response"), list):
            eval_input["response"] = _preprocess_messages(eval_input["response"])
        if isinstance(eval_input.get("query"), list):
            eval_input["query"] = _preprocess_messages(eval_input["query"])

        result = await self._flow(timeout=self._LLM_CALL_TIMEOUT, **eval_input)
        llm_output = result.get("llm_output", result) if isinstance(result, dict) else result

        score = math.nan
        llm_output_is_dict = isinstance(llm_output, dict)
        if llm_output_is_dict or isinstance(llm_output, str):
            reason = ""
            if llm_output_is_dict:
                score = float(llm_output.get("score", math.nan))
                reason = llm_output.get("explanation", "")
            else:
                score, reason = parse_quality_evaluator_reason_score(llm_output, valid_score_range="[1-5]")

            binary_result = self._get_binary_result(score)

            input_token_count = result.get("input_token_count", 0) if isinstance(result, dict) else 0
            output_token_count = result.get("output_token_count", 0) if isinstance(result, dict) else 0
            total_token_count = result.get("total_token_count", 0) if isinstance(result, dict) else 0
            finish_reason = result.get("finish_reason", "") if isinstance(result, dict) else ""
            model_id = result.get("model_id", "") if isinstance(result, dict) else ""
            sample_input = result.get("sample_input", "") if isinstance(result, dict) else ""
            sample_output = result.get("sample_output", "") if isinstance(result, dict) else ""

            # updating the result key and threshold to int based on the schema
            return {
                f"{self._result_key}": int(score),
                f"{self._result_key}_result": binary_result,
                f"{self._result_key}_threshold": int(self._threshold),
                f"{self._result_key}_reason": reason,
                f"{self._result_key}_prompt_tokens": input_token_count,
                f"{self._result_key}_completion_tokens": output_token_count,
                f"{self._result_key}_total_tokens": total_token_count,
                f"{self._result_key}_finish_reason": finish_reason,
                f"{self._result_key}_model": model_id,
                f"{self._result_key}_sample_input": sample_input,
                f"{self._result_key}_sample_output": sample_output,
            }

        if logger:
            logger.warning("LLM output is not a dictionary, returning NaN for the score.")

        binary_result = self._get_binary_result(score)
        return {
            self._result_key: float(score),
            f"{self._result_key}_result": binary_result,
            f"{self._result_key}_threshold": self._threshold,
        }
