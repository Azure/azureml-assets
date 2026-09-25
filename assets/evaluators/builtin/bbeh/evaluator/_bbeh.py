# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
BBEH (BIG-Bench Extra Hard) Evaluator.

This evaluator implements the fuzzy matching logic from Google DeepMind's BBEH benchmark
for evaluating model responses on reasoning tasks. It extracts answers from model output
and compares them against ground truth with tolerance for common variations.

Reference: https://github.com/google-deepmind/bbeh
"""

import json
import logging
from typing import Dict, List
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
            return message_content
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
        raise EvaluationException(
            message="messages must be provided as a non-empty list of message dictionaries.",
            blame=ErrorBlame.USER_ERROR,
            category=ErrorCategory.INVALID_VALUE,
            target=ErrorTarget.EVALUATE,
        )
    return _parse_response_for_evaluation(messages)


def strip_latex(response: str) -> str:
    r"""Strip LaTeX formatting from a response.

    Handles common LaTeX patterns used in math answers:
    - Dollar signs: $...$
    - Boxed: \boxed{...}
    - Text: \text{...}
    - Texttt: \texttt{...}

    :param response: The response text potentially containing LaTeX.
    :type response: str
    :return: The response with LaTeX formatting stripped.
    :rtype: str
    """
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def extract_answer(sample: str) -> str:
    """Extract the final answer from a model response.

    Looks for common answer prefixes and extracts the text following them.
    Also strips trailing periods and LaTeX formatting.

    :param sample: The full model response.
    :type sample: str
    :return: The extracted answer portion.
    :rtype: str
    """
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return strip_latex(answer)


def fuzzy_match(prediction: str, reference: str) -> bool:
    """Fuzzy match function for BBEH evaluation.

    Handles common variations between model predictions and reference answers:
    - Exact string match
    - Parenthesized vs bare: (a) matches a
    - Numeric equality: 4.0 matches 4
    - Quote normalization: removes single quotes
    - Bracket variations: [answer] matches answer
    - Question mark endings: answer? matches answer

    :param prediction: The preprocessed model prediction.
    :type prediction: str
    :param reference: The preprocessed reference answer.
    :type reference: str
    :return: True if the prediction matches the reference, False otherwise.
    :rtype: bool
    """
    if prediction == reference:
        return True

    # (a) vs a - handle parenthesized single characters
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction

    # Numbers - compare as floats
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass

    # Quote issues - ignore single quotes
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True

    # Bracket issues - handle [answer] vs answer
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True

    # Question mark issues - handle trailing question marks
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True

    return False


def preprocess_sample(sample: str) -> str:
    """Preprocess model output for comparison.

    Applies normalization:
    - Strips whitespace
    - Extracts answer portion
    - Converts to lowercase
    - Normalizes comma spacing
    - Removes markdown bold markers
    - Takes only the first line
    - Removes trailing periods

    :param sample: The raw model output.
    :type sample: str
    :return: The preprocessed prediction.
    :rtype: str
    """
    prediction = extract_answer(sample.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction


def preprocess_reference(reference: str) -> str:
    """Preprocess reference answer for comparison.

    Applies normalization:
    - Strips whitespace
    - Converts to lowercase
    - Normalizes comma spacing

    :param reference: The reference answer.
    :type reference: str
    :return: The preprocessed reference.
    :rtype: str
    """
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference


def evaluate_correctness(sample: str, reference: str) -> bool:
    """Evaluate if a model sample correctly answers the reference.

    Main evaluation function that preprocesses both inputs and applies
    fuzzy matching.

    :param sample: The raw model output.
    :type sample: str
    :param reference: The expected answer.
    :type reference: str
    :return: True if the sample is correct, False otherwise.
    :rtype: bool
    """
    prediction = preprocess_sample(sample)
    reference = preprocess_reference(reference)
    return fuzzy_match(prediction, reference)


class BBEHEvaluator(EvaluatorBase):
    r"""Evaluator for BIG-Bench Extra Hard (BBEH) benchmark.

    This evaluator implements the official BBEH fuzzy matching logic from
    Google DeepMind for evaluating model responses on challenging reasoning
    tasks. It handles answer extraction and comparison with tolerance for
    common formatting variations.

    The evaluator:
    1. Extracts the answer from model output (handles "The answer is:" prefixes)
    2. Strips LaTeX formatting (\boxed{}, \text{}, etc.)
    3. Normalizes both prediction and reference (lowercase, spacing)
    4. Applies fuzzy matching (handles (a) vs a, numeric equality, quotes, etc.)

    :return: A dictionary with the evaluation results.
    :rtype: Dict[str, any]

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation import BBEHEvaluator

            evaluator = BBEHEvaluator()

            result = evaluator(
                response="Let me think... The final answer is: \\\\boxed{4}.",
                ground_truth="4"
            )
            # result: {"bbeh": True, "bbeh_result": "pass"}

    .. admonition:: Example with parenthesized answer:

        .. code-block:: python

            result = evaluator(
                response="Ok The answer is: (A)",
                ground_truth="a"
            )
            # result: {"bbeh": True, "bbeh_result": "pass"}
    """

    id = "azureai://built-in/evaluators/bbeh"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(self):
        """Initialize the BBEH evaluator."""
        super().__init__()
        logger.debug("BBEHEvaluator initialized")

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, any]:
        """Produce a BBEH evaluation result.

        Evaluates whether the model response correctly answers the question
        using fuzzy matching logic.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result with score and match information.
        :rtype: Dict
        """
        response = _parse_response_for_evaluation(eval_input.get("response", ""))
        ground_truth = eval_input.get("ground_truth", "")

        if not response:
            logger.debug("Empty response, returning False")
            return {
                "bbeh": False,
                "bbeh_result": EVALUATION_PASS_FAIL_MAPPING[False],
            }

        if not ground_truth:
            logger.warning("Empty ground_truth provided")
            return {
                "bbeh": False,
                "bbeh_result": EVALUATION_PASS_FAIL_MAPPING[False],
            }

        # Evaluate correctness using BBEH fuzzy matching
        is_correct = evaluate_correctness(response, ground_truth)

        logger.debug(
            "BBEH evaluation: is_correct=%s, response_length=%d",
            is_correct,
            len(response),
        )

        return {
            "bbeh": is_correct,
            "bbeh_result": EVALUATION_PASS_FAIL_MAPPING[is_correct],
        }

    @override
    async def _real_call(self, **kwargs):
        messages = kwargs.pop("messages", None)
        if messages is not None:
            kwargs["response"] = _response_from_messages(messages)
        return await super()._real_call(**kwargs)

    @overload  # type: ignore
    def __call__(self, *, response: str, ground_truth: str) -> Dict[str, any]:
        """
        Evaluate whether the response correctly answers the BBEH question.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword ground_truth: The expected answer.
        :paramtype ground_truth: str
        :return: The evaluation result containing score and pass/fail information.
        :rtype: Dict[str, any]
        """

    @overload
    def __call__(self, *, messages: List[dict], ground_truth: str) -> Dict[str, any]:
        """Evaluate BBEH using the final agent response in messages."""

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate whether the response correctly answers the BBEH question.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword ground_truth: The expected answer.
        :paramtype ground_truth: str
        :return: The evaluation result containing score and pass/fail information.
        :rtype: Dict[str, any]
        """
        return super().__call__(*args, **kwargs)
