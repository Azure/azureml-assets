# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import math
import os
import re
from typing import Dict, Union

from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common import PromptyEvaluatorBase
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING

# ---------------------------------------------------------------------------
# Imports target azure-ai-evaluation >= 1.18.1. Each ``except ImportError``
# branch below inlines the corresponding azure-ai-evaluation 1.18.1
# implementation so the evaluator also runs on azure-ai-evaluation 1.17.x,
# which predates these symbols. The 1.17.x compatibility branches are kept only
# for backward compatibility and can be removed once 1.17.x is no longer
# supported.
# ---------------------------------------------------------------------------

from azure.ai.evaluation._common.constants import PROMPT_BASED_REASON_EVALUATORS
from azure.ai.evaluation._common.utils import parse_quality_evaluator_reason_score

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _is_intermediate_response, _preprocess_messages
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (
        _is_intermediate_response,
        _preprocess_messages,
    )

# Re-exported so the module keeps exposing the message-preprocessing helpers used
# by the test suite; they are invoked indirectly through _preprocess_messages.
try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import (  # noqa: F401
        _drop_mcp_approval_messages,
        _normalize_function_call_types,
    )

logger = logging.getLogger(__name__)


class SimilarityEvaluator(PromptyEvaluatorBase):
    """
    Evaluates similarity score for a given query, response, and ground truth.

    The similarity measure evaluates the likeness between a ground truth sentence (or document) and the
    AI model's generated prediction. This calculation involves creating sentence-level embeddings for both
    the ground truth and the model's prediction, which are high-dimensional vector representations capturing
    the semantic meaning and context of the sentences.

    Use it when you want an objective evaluation of an AI model's performance, particularly in text generation
    tasks where you have access to ground truth responses. Similarity enables you to assess the generated
    text's semantic alignment with the desired content, helping to gauge the model's quality and accuracy.

    Similarity scores range from 1 to 5, with 1 being the least similar and 5 being the most similar.

    :param model_config: Configuration for the Azure OpenAI model.
    :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
        ~azure.ai.evaluation.OpenAIModelConfiguration]
    :param threshold: The threshold for the similarity evaluator. Default is 3.
    :type threshold: int
    :param credential: The credential for authenticating to Azure AI service.
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword is_reasoning_model: If True, the evaluator will use reasoning model configuration (o1/o3 models).
        This will adjust parameters like max_completion_tokens and remove unsupported parameters. Default is False.
    :paramtype is_reasoning_model: bool

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START similarity_evaluator]
            :end-before: [END similarity_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a SimilarityEvaluator with a four-gram rouge type.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START similarity_evaluator]
            :end-before: [END similarity_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call SimilarityEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_similarity_evaluator]
            :end-before: [END threshold_similarity_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with a threshold and call a SimilarityEvaluator.

    .. note::

        To align with our support of a diverse set of models, an output key without the `gpt_` prefix has been added.
        To maintain backwards compatibility, the old key with the `gpt_` prefix is still be present in the output;
        however, it is recommended to use the new key moving forward as the old key will be deprecated in the future.
    """

    # Constants must be defined within eval's directory to be save/loadable

    _PROMPTY_FILE = "similarity.prompty"
    _RESULT_KEY = "similarity"

    id = "azureai://built-in/evaluators/similarity"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, model_config, *, threshold=3, credential=None, **kwargs):
        """Initialize the Similarity evaluator.

        :param model_config: Configuration for the Azure OpenAI model.
        :type model_config: Union[~azure.ai.evaluation.AzureOpenAIModelConfiguration,
            ~azure.ai.evaluation.OpenAIModelConfiguration]
        :param threshold: The threshold for evaluation.
        :type threshold: int
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

    # Ignoring a mypy error about having only 1 overload function.
    # We want to use the overload style for all evals, even single-inputs. This is both to make
    # refactoring to multi-input styles easier, stylistic consistency consistency across evals,
    # and due to the fact that non-overloaded syntax now causes various parsing issues that
    # we don't want to deal with.
    @overload  # type: ignore
    def __call__(self, *, query: str, response: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate similarity.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be evaluated.
        :paramtype ground_truth: str
        :return: The similarity score.
        :rtype: Dict[str, float]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate similarity.

        :keyword query: The query to be evaluated.
        :paramtype query: str
        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be evaluated.
        :paramtype ground_truth: str
        :return: The similarity score.
        :rtype: Dict[str, float]
        """
        return super().__call__(*args, **kwargs)

    @override
    def _convert_kwargs_to_eval_input(self, **kwargs):
        """Convert keyword arguments to evaluation input, with validation."""
        conversation = kwargs.get("conversation")
        if conversation is not None:
            return super()._convert_kwargs_to_eval_input(**kwargs)

        query = kwargs.get("query")
        response = kwargs.get("response")
        ground_truth = kwargs.get("ground_truth")

        # Validate required fields are not None
        if query is None:
            raise EvaluationException(
                message="Either 'conversation' or individual inputs must be provided. 'query' is missing.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.SIMILARITY_EVALUATOR,
            )

        if response is None:
            raise EvaluationException(
                message="Either 'conversation' or individual inputs must be provided. 'response' is missing.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.SIMILARITY_EVALUATOR,
            )

        if ground_truth is None:
            raise EvaluationException(
                message="Either 'conversation' or individual inputs must be provided. 'ground_truth' is missing.",
                blame=ErrorBlame.USER_ERROR,
                category=ErrorCategory.MISSING_FIELD,
                target=ErrorTarget.SIMILARITY_EVALUATOR,
            )

        return super()._convert_kwargs_to_eval_input(**kwargs)

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

    @override
    async def _do_eval(self, eval_input: Dict):  # type: ignore[override]
        """Do a similarity evaluation.

        :param eval_input: The input to the evaluator.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        result = await self._the_super_do_eval(eval_input)
        # Check if base returned nan (invalid output case); None means not-applicable/skipped
        _score = result.get(self._result_key, 0)
        if _score is not None and math.isnan(_score):
            raise EvaluationException(
                message="Evaluator returned invalid output.",
                blame=ErrorBlame.SYSTEM_ERROR,
                category=ErrorCategory.FAILED_EXECUTION,
                target=ErrorTarget.SIMILARITY_EVALUATOR,
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
