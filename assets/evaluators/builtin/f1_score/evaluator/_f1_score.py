# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
from collections import Counter
from typing import List, Dict
from typing_extensions import overload, override

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


class F1ScoreEvaluator(EvaluatorBase):
    """
    Calculates the F1 score for a given response and ground truth or a multi-turn conversation.

    F1 Scores range from 0 to 1, with 1 being the best possible score.

    The F1-score computes the ratio of the number of shared words between the model generation and
    the ground truth. Ratio is computed over the individual words in the generated response against those in the ground
    truth answer. The number of shared words between the generation and the truth is the basis of the F1 score:
    precision is the ratio of the number of shared words to the total number of words in the generation, and recall
    is the ratio of the number of shared words to the total number of words in the ground truth.

    Use the F1 score when you want a single comprehensive metric that combines both recall and precision in your
    model's responses. It provides a balanced evaluation of your model's performance in terms of capturing accurate
    information in the response.

    :param threshold: The threshold for the F1 score evaluator. Default is 0.5.
    :type threshold: float

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START f1_score_evaluator]
            :end-before: [END f1_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call an F1ScoreEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START f1_score_evaluator]
            :end-before: [END f1_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call F1ScoreEvaluator using Azure AI Project URL in following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_f1_score_evaluator]
            :end-before: [END threshold_f1_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call an F1ScoreEvaluator.
    """

    id = "azureai://built-in/evaluators/f1_score"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(self, *, threshold=0.5):
        """Initialize the F1 Score evaluator.

        :param threshold: The threshold for evaluation.
        :type threshold: float
        """
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(threshold=threshold, _higher_is_better=self._higher_is_better)

    @classmethod
    def _compute_f1_score(cls, response: str, ground_truth: str) -> float:
        import re
        import string

        class QASplitTokenizer:
            """Quality assurance tokenizer that splits text on whitespace."""

            def __call__(self, line) -> List[str]:
                """Tokenize an input line using split() on whitespace.

                :param line: The input segment to be tokenized
                :type line: str
                :return: The tokenized segment
                :rtype: List[str]
                """
                return line.split()

        def normalize_text(text: str) -> str:
            """Lower text and remove punctuation, articles and extra whitespace.

            :param text: The text to be normalized
            :type text: str
            :return: The normalized text
            :rtype: str
            """

            def remove_articles(text):
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text):
                return " ".join(text.split())

            def remove_punctuation(text):
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punctuation(lower(text))))

        tokenizer = QASplitTokenizer()
        prediction_tokens = tokenizer(normalize_text(response))
        reference_tokens = tokenizer(normalize_text(ground_truth))

        common_tokens = Counter(prediction_tokens) & Counter(reference_tokens)
        num_common_tokens = sum(common_tokens.values())

        if num_common_tokens == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_common_tokens / len(prediction_tokens)
            recall = 1.0 * num_common_tokens / len(reference_tokens)

            f1 = (2.0 * precision * recall) / (precision + recall)

        return f1

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, float]:
        """Produce an f1 score evaluation result.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        ground_truth = eval_input["ground_truth"]
        response = _parse_response_for_evaluation(eval_input["response"])
        # Run f1 score computation.
        f1_result = self._compute_f1_score(response=response, ground_truth=ground_truth)
        binary_result = False
        if self._higher_is_better:
            if f1_result >= self._threshold:
                binary_result = True
        else:
            if f1_result <= self._threshold:
                binary_result = True
        return {
            "f1_score": f1_result,
            "f1_score_score": f1_result,
            "f1_score_passed": binary_result,
            "f1_score_result": EVALUATION_PASS_FAIL_MAPPING[binary_result],
            "f1_score_reason": None,
            "f1_score_status": "completed",
            "f1_score_threshold": self._threshold,
            "f1_score_properties": None,
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
    def __call__(self, *, response: str, ground_truth: str) -> Dict[str, float]:
        """
        Evaluate F1 score.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be evaluated.
        :paramtype ground_truth: str
        :return: The F1 score.
        :rtype: Dict[str, float]
        """

    @overload
    def __call__(self, *, messages: List[dict], ground_truth: str) -> Dict[str, float]:
        """Evaluate F1 score using the final agent response in messages."""

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate F1 score.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be evaluated.
        :paramtype ground_truth: str
        :return: The F1 score.
        :rtype: Dict[str, float]
        """
        return super().__call__(*args, **kwargs)
