# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from typing_extensions import overload, override

from azure.ai.evaluation._common.utils import nltk_tokenize

from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
import logging
from azure.ai.evaluation._exceptions import EvaluationException, ErrorCategory, ErrorTarget



logger = logging.getLogger(__name__)
class BleuScoreEvaluator(EvaluatorBase):
    """
    Calculate the BLEU score for a given response and ground truth.

    BLEU (Bilingual Evaluation Understudy) score is commonly used in natural language processing (NLP) and machine
    translation. It is widely used in text summarization and text generation use cases.

    Use the BLEU score when you want to evaluate the similarity between the generated text and reference text,
    especially in tasks such as machine translation or text summarization, where n-gram overlap is a significant
    indicator of quality.

    The BLEU score ranges from 0 to 1, with higher scores indicating better quality.
    :param threshold: The threshold for the evaluation. Default is 0.5.
    :type threshold: float

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START bleu_score_evaluator]
            :end-before: [END bleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call an BleuScoreEvaluator using azure.ai.evaluation.AzureAIProject

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START bleu_score_evaluator]
            :end-before: [END bleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call an BleuScoreEvaluator using Azure AI Project URL in following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}

    .. admonition:: Example with Threshold:
        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_bleu_score_evaluator]
            :end-before: [END threshold_bleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call an BleuScoreEvaluator.
    """

    id = "azureai://built-in/evaluators/bleu_score"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    # region Vendored base helpers (copied from azure-sdk-for-python PR #46436)
    # The following methods are inlined copies of helpers from
    # azure.ai.evaluation._evaluators._common._base_eval / _base_prompty_eval.
    # They are vendored here because the runtime environment ships an older
    # version of those base files. Do not modify without re-syncing with
    # upstream PR #46436.

    async def _real_call(self, **kwargs):
        """The asynchronous call where real end-to-end evaluation logic is performed.

        :keyword kwargs: The inputs to evaluate.
        :type kwargs: Dict
        :return: The evaluation result.
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

    # endregion

    def __init__(self, *, threshold=0.5):
        """Initialize the BLEU Score evaluator.

        :param threshold: The threshold for evaluation.
        :type threshold: float
        """
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(threshold=threshold, _higher_is_better=self._higher_is_better)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, float]:
        """Produce a bleu score evaluation result.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        ground_truth = eval_input["ground_truth"]
        response = eval_input["response"]
        reference_tokens = nltk_tokenize(ground_truth)
        hypothesis_tokens = nltk_tokenize(response)

        # NIST Smoothing
        smoothing_function = SmoothingFunction().method4
        score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smoothing_function)
        binary_result = False
        if self._higher_is_better:
            binary_result = score >= self._threshold
        else:
            binary_result = score <= self._threshold

        return {
            "bleu": score,
            "bleu_score": score,
            "bleu_passed": binary_result,
            "bleu_result": EVALUATION_PASS_FAIL_MAPPING[binary_result],
            "bleu_reason": None,
            "bleu_status": "completed",
            "bleu_threshold": self._threshold,
            "bleu_properties": None,
        }

    @overload  # type: ignore
    def __call__(self, *, response: str, ground_truth: str):
        """
        Evaluate the BLEU score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The BLEU score.
        :rtype: Dict[str, float]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate the BLEU score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The BLEU score.
        :rtype: Dict[str, float]
        """
        return super().__call__(*args, **kwargs)
