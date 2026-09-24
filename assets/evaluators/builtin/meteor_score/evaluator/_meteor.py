# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from typing import Dict, List

from nltk.translate.meteor_score import meteor_score
from typing_extensions import overload, override

from azure.ai.evaluation._common.utils import nltk_tokenize, ensure_nltk_data_downloaded
from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget

try:  # azure-ai-evaluation >= 1.18.1
    from azure.ai.evaluation._common.utils import _preprocess_messages
except ImportError:  # azure-ai-evaluation 1.17.x (backward compat; remove when 1.17.x is dropped)  # pragma: no cover
    from azure.ai.evaluation._evaluators._common._base_prompty_eval import _preprocess_messages

logger = logging.getLogger(__name__)


def _extract_final_text_response(messages):
    """Extract only the latest assistant text message, dropping tool calls/results.

    Scans messages in reverse order for the latest assistant text, stopping as
    soon as a user message is reached so text from earlier turns is never used.

    :param messages: The preprocessed list of chat-message dicts.
    :type messages: list
    :return: The latest assistant text, or an empty string if none is found
        before the latest user message.
    :rtype: str
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            break
        if role != "assistant":
            continue
        message_content = msg.get("content", []) or []
        if isinstance(message_content, str):
            if message_content:
                return message_content
            continue
        text_lines = [
            content["text"]
            for content in message_content
            if isinstance(content, dict) and content.get("text")
        ]
        if text_lines:
            return "\n".join(text_lines)
    return ""


def _parse_response_for_evaluation(response):
    """Flatten ``response`` into a plain string, parsing a JSON-encoded list of messages if needed.

    If ``response`` is a string, attempt to ``json.loads`` it. When it (or an already-parsed
    ``response``) is a list of chat-message dicts, only the plain text of the latest assistant
    message is extracted (tool calls, tool results, and other message types are dropped); an
    empty string is returned if no such text is found. Any other input (a plain string that is
    not JSON, or a JSON value that isn't a list) is returned unchanged.

    :param response: The raw response value from the eval input.
    :type response: Any
    :return: A plain string ready for deterministic scoring.
    :rtype: str
    """
    parsed = response
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except (ValueError, TypeError):
            return response
    if isinstance(parsed, list):
        try:
            messages = _preprocess_messages(parsed)
            return _extract_final_text_response(messages)
        except Exception:
            logger.debug("Could not extract plain text from response messages; treating as empty response")
            return ""
    return response


def _response_from_messages(messages):
    """Extract the final agent text response from a list of chat messages."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be provided as a non-empty list of message dictionaries.")
    return _parse_response_for_evaluation(messages)


class MeteorScoreEvaluator(EvaluatorBase):
    """
    Calculates the METEOR score for a given response and ground truth.

    The METEOR (Metric for Evaluation of Translation with Explicit Ordering) score grader evaluates generated text by
    comparing it to reference texts, focusing on precision, recall, and content alignment. It addresses limitations of
    other metrics like BLEU by considering synonyms, stemming, and paraphrasing. METEOR score considers synonyms and
    word stems to more accurately capture meaning and language variations. In addition to machine translation and
    text summarization, paraphrase detection is an optimal use case for the METEOR score.

    Use the METEOR score when you want a more linguistically informed evaluation metric that captures not only
    n-gram overlap but also accounts for synonyms, stemming, and word order. This is particularly useful for evaluating
    tasks like machine translation, text summarization, and text generation.

    The METEOR score ranges from 0 to 1, with 1 indicating a perfect match.

    :param alpha: The METEOR score alpha parameter. Default is 0.9.
    :type alpha: float
    :param beta: The METEOR score beta parameter. Default is 3.0.
    :type beta: float
    :param gamma: The METEOR score gamma parameter. Default is 0.5.
    :type gamma: float
    :param threshold: The threshold for the METEOR score evaluator. Default is 0.5.
    :type threshold: float

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START meteor_score_evaluator]
            :end-before: [END meteor_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a MeteorScoreEvaluator with alpha of 0.8.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START meteor_score_evaluator]
            :end-before: [END meteor_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call MeteorScoreEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_meteor_score_evaluator]
            :end-before: [END threshold_meteor_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a MeteorScoreEvaluator.
    """

    id = "azureai://built-in/evaluators/meteor_score"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5, *, threshold: float = 0.5):
        """Initialize the METEOR Score evaluator.

        :param alpha: Alpha parameter for METEOR score calculation.
        :type alpha: float
        :param beta: Beta parameter for METEOR score calculation.
        :type beta: float
        :param gamma: Gamma parameter for METEOR score calculation.
        :type gamma: float
        :param threshold: The threshold for evaluation.
        :type threshold: float
        """
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        ensure_nltk_data_downloaded()
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(threshold=threshold, _higher_is_better=self._higher_is_better)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, float]:
        """Produce a meteor score evaluation result.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        ground_truth = eval_input["ground_truth"]
        response = _parse_response_for_evaluation(eval_input["response"])
        reference_tokens = nltk_tokenize(ground_truth)
        hypothesis_tokens = nltk_tokenize(response)
        score = meteor_score(
            [reference_tokens],
            hypothesis_tokens,
            alpha=self._alpha,
            beta=self._beta,
            gamma=self._gamma,
        )
        binary_result = False
        if self._higher_is_better:
            if score >= self._threshold:
                binary_result = True
        else:
            if score <= self._threshold:
                binary_result = True
        return {
            "meteor": score,
            "meteor_score": score,
            "meteor_passed": binary_result,
            "meteor_result": EVALUATION_PASS_FAIL_MAPPING[binary_result],
            "meteor_reason": None,
            "meteor_status": "completed",
            "meteor_threshold": self._threshold,
            "meteor_properties": None,
        }

    @override
    async def _real_call(self, **kwargs):
        """Perform the asynchronous call where real end-to-end evaluation logic runs.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
        :rtype: Union[DoEvalResult[T_EvalValue], AggregateResult[T_EvalValue]]
        """
        messages = kwargs.pop("messages", None)
        if messages is not None:
            kwargs["response"] = _response_from_messages(messages)

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

    @overload  # type: ignore
    def __call__(self, *, ground_truth: str, response: str) -> Dict[str, float]:
        """
        Evaluate the METEOR score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The METEOR score.
        :rtype: Dict[str, float]
        """

    @overload
    def __call__(self, *, ground_truth: str, messages: List[dict]) -> Dict[str, float]:
        """Evaluate METEOR using the final agent response in messages."""

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate the METEOR score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The METEOR score.
        :rtype: Dict[str, float]
        """
        return super().__call__(*args, **kwargs)
