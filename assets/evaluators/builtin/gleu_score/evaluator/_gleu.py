# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from typing import Dict
from nltk.translate.gleu_score import sentence_gleu
from typing_extensions import overload, override

from azure.ai.evaluation._common.utils import nltk_tokenize

from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING
from azure.ai.evaluation._exceptions import EvaluationException, ErrorBlame, ErrorCategory, ErrorTarget

logger = logging.getLogger(__name__)


class GleuScoreEvaluator(EvaluatorBase):
    """
    Calculates the GLEU (Google-BLEU) score between a response and the ground truth.

    The GLEU (Google-BLEU) score evaluator measures the similarity between generated and reference texts by
    evaluating n-gram overlap, considering both precision and recall. This balanced evaluation, designed for
    sentence-level assessment, makes it ideal for detailed analysis of translation quality. GLEU is well-suited for
    use cases such as machine translation, text summarization, and text generation.

    GLEU scores range from 0 to 1, where a value of 1 represents perfect overlap between the response and
    the ground truth and a value of 0 indicates no overlap.

    :param threshold: The threshold for the GLEU evaluator. Default is 0.5.
    :type threshold: float

    .. admonition:: Example:

        .. literalinclude:: ../samples/evaluation_samples_evaluate.py
            :start-after: [START gleu_score_evaluator]
            :end-before: [END gleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call a GleuScoreEvaluator.

    .. admonition:: Example with Threshold:

        .. literalinclude:: ../samples/evaluation_samples_threshold.py
            :start-after: [START threshold_gleu_score_evaluator]
            :end-before: [END threshold_gleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize with threshold and call a GleuScoreEvaluator.

    .. admonition:: Example using Azure AI Project URL:

        .. literalinclude:: ../samples/evaluation_samples_evaluate_fdp.py
            :start-after: [START gleu_score_evaluator]
            :end-before: [END gleu_score_evaluator]
            :language: python
            :dedent: 8
            :caption: Initialize and call GleuScoreEvaluator using Azure AI Project URL in the following format
                https://{resource_name}.services.ai.azure.com/api/projects/{project_name}
    """

    id = "azureai://built-in/evaluators/gleu_score"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    @override
    def __init__(self, *, threshold=0.5):
        """Initialize the GLEU Score evaluator.

        :param threshold: The threshold for evaluation.
        :type threshold: float
        """
        self._threshold = threshold
        self._higher_is_better = True
        super().__init__(threshold=threshold, _higher_is_better=self._higher_is_better)

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, float]:
        """Produce a glue score evaluation result.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result.
        :rtype: Dict
        """
        ground_truth = eval_input["ground_truth"]
        response = eval_input["response"]
        reference_tokens = nltk_tokenize(ground_truth)
        hypothesis_tokens = nltk_tokenize(response)

        score = sentence_gleu([reference_tokens], hypothesis_tokens)
        binary_result = False
        if self._higher_is_better:
            if score >= self._threshold:
                binary_result = True
        else:
            if score <= self._threshold:
                binary_result = True
        return {
            "gleu": score,
            "gleu_score": score,
            "gleu_passed": binary_result,
            "gleu_result": EVALUATION_PASS_FAIL_MAPPING[binary_result],
            "gleu_reason": None,
            "gleu_status": "completed",
            "gleu_threshold": self._threshold,
            "gleu_properties": None,
        }

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

    @overload  # type: ignore
    def __call__(self, *, ground_truth: str, response: str):
        """
        Evaluate the GLEU score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The GLEU score.
        :rtype: Dict[str, float]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate the GLEU score between the response and the ground truth.

        :keyword response: The response to be evaluated.
        :paramtype response: str
        :keyword ground_truth: The ground truth to be compared against.
        :paramtype ground_truth: str
        :return: The GLEU score.
        :rtype: Dict[str, float]
        """
        return super().__call__(*args, **kwargs)
