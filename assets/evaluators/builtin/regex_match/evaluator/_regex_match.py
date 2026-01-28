# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Dict, List, Optional, Union
from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING


class RegexMatchEvaluator(EvaluatorBase):
    """
    Extracts a value from the model response using regex and compares it against ground truth.

    This evaluator applies regex pattern(s) ONLY to the model response to extract an answer,
    then compares the extracted value directly to the ground truth (which is treated as a
    plain value, not processed by regex).

    This aligns with BabelBench GPQA semantics where:
    - Regex is used for extraction from model response only
    - Ground truth is a plain expected value (e.g., "A", "B", "C", "D")
    - Scoring is binary: 1.0 if extracted value matches ground truth, 0.0 otherwise

    :param patterns: One or more regex patterns to try for extraction. Each pattern must
        contain at least one capture group. Patterns are tried sequentially and the first
        successful extraction is used.
    :type patterns: Union[str, List[str]]
    :param ignore_case: If True, perform case-insensitive comparison of extracted value
        against ground truth. Default is True.
    :type ignore_case: bool

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation import RegexMatchEvaluator

            # Create evaluator with GPQA pattern
            evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\\s*:\\s*([A-D])")

            # Evaluate a response against ground truth
            result = evaluator(
                response="Based on my analysis, ANSWER: B is the correct choice.",
                ground_truth="B"
            )
            # result: {"regex_match": 1.0, "regex_match_result": "pass",
            #          "extracted_value": "B"}

    .. admonition:: Example with multiple patterns:

        .. code-block:: python

            # Create evaluator with multiple patterns (tried sequentially)
            evaluator = RegexMatchEvaluator(
                patterns=[
                    r"(?i)ANSWER\\s*:\\s*([A-D])",
                    r"(?i)The answer is\\s*([A-D])",
                    r"\\b([A-D])\\b"
                ]
            )

            result = evaluator(
                response="I think the answer is C.",
                ground_truth="C"
            )
            # result: {"regex_match": 1.0, "regex_match_result": "pass",
            #          "extracted_value": "C"}
    """

    id = "azureai://built-in/evaluators/regex_match"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(
        self,
        *,
        patterns: Union[str, List[str]],
        ignore_case: bool = True
    ):
        """Initialize the Regex Match evaluator.

        :param patterns: One or more regex patterns to try for extraction. Each pattern
            must contain at least one capture group. Patterns are tried sequentially and
            the first successful extraction is used.
        :type patterns: Union[str, List[str]]
        :param ignore_case: If True, perform case-insensitive comparison of extracted
            value against ground truth. Default is True.
        :type ignore_case: bool
        :raises ValueError: If any pattern is empty or does not contain a capture group.
        """
        # Normalize patterns to a list
        if isinstance(patterns, str):
            patterns = [patterns]

        if not patterns:
            raise ValueError("At least one pattern must be provided.")

        # Validate and compile all patterns
        compiled_patterns = []
        for i, pattern in enumerate(patterns):
            if not pattern:
                raise ValueError(f"Pattern at index {i} must not be empty.")

            try:
                compiled = re.compile(pattern)
                if compiled.groups < 1:
                    raise ValueError(
                        f"Pattern at index {i} must contain at least one capture group. "
                        "Use parentheses to define a capture group, e.g., '(\\d+)'."
                    )
                compiled_patterns.append(compiled)
            except re.error as e:
                raise ValueError(f"Invalid regular expression pattern at index {i}: {e}")

        self._patterns = patterns
        self._compiled_patterns = compiled_patterns
        self._ignore_case = ignore_case
        super().__init__()

    def _extract_from_response(self, response: str) -> Optional[str]:
        """Extract value from response using patterns sequentially.

        Tries each pattern in order and returns the first successful extraction.

        :param response: The model response text to extract from.
        :type response: str
        :return: The first captured group from the first matching pattern, or None.
        :rtype: Optional[str]
        """
        if not response:
            return None

        for compiled_pattern in self._compiled_patterns:
            match = compiled_pattern.search(response)
            if match and match.groups():
                return match.group(1)

        return None

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, any]:
        """Produce a regex match evaluation result.

        Extracts value from response using regex, compares to ground truth.
        Ground truth is treated as a plain value (no regex applied).

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result with score and extracted value.
        :rtype: Dict
        """
        ground_truth = eval_input.get("ground_truth", "")
        response = eval_input.get("response", "")

        # Extract value from response only (not from ground_truth)
        extracted_value = self._extract_from_response(response)

        # Determine score
        if extracted_value is None:
            # No match found in response -> score = 0
            score = 0.0
        elif not ground_truth:
            # No ground truth provided -> score = 0
            score = 0.0
        else:
            # Compare extracted value to ground truth
            if self._ignore_case:
                is_match = extracted_value.lower() == ground_truth.lower()
            else:
                is_match = extracted_value == ground_truth
            score = 1.0 if is_match else 0.0

        return {
            "regex_match": score,
            "regex_match_result": EVALUATION_PASS_FAIL_MAPPING[score == 1.0],
            "extracted_value": extracted_value,
        }

    @overload
    def __call__(self, *, response: str, ground_truth: str) -> Dict[str, any]:
        """
        Evaluate regex match between extracted response value and ground truth.

        :keyword response: The model response to extract value from using regex.
        :paramtype response: str
        :keyword ground_truth: The expected value to compare against (plain text, no regex).
        :paramtype ground_truth: str
        :return: The evaluation result containing score and extracted value.
        :rtype: Dict[str, any]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate regex match between extracted response value and ground truth.

        :keyword response: The model response to extract value from using regex.
        :paramtype response: str
        :keyword ground_truth: The expected value to compare against (plain text, no regex).
        :paramtype ground_truth: str
        :return: The evaluation result containing score and extracted value.
        :rtype: Dict[str, any]
        """
        return super().__call__(*args, **kwargs)
