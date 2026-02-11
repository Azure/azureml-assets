# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from typing import Dict, List, Optional, Union
from typing_extensions import overload, override

from azure.ai.evaluation._evaluators._common import EvaluatorBase
from azure.ai.evaluation._constants import EVALUATION_PASS_FAIL_MAPPING


# Regex to detect {{ground_truth}} reference in patterns
GROUND_TRUTH_PATTERN = re.compile(r"\{\{ground_truth\}\}")


class RegexMatchEvaluator(EvaluatorBase):
    r"""
    Evaluates whether a model response matches one or more regex patterns.

    This evaluator applies regex pattern(s) to the model response text and returns
    a binary accuracy score based on whether any pattern matches. Patterns can include
    a reference to {{ground_truth}} that is resolved per-row from the evaluation input.

    This is designed for benchmarking tasks where correctness is determined by pattern
    matching, such as GPQA where the expected answer is embedded in the regex pattern.

    Key characteristics:
    - Regex is applied ONLY to the model response text
    - Matching is existence-based (does the pattern match?), not extraction-based
    - Scoring is binary: 1.0 if any pattern matches, 0.0 otherwise
    - No capture groups required - simple pattern matching
    - Static patterns are compiled once; dynamic patterns (with {{ground_truth}}) are resolved per-row

    :param patterns: One or more regex patterns to match against the response. Patterns
        are tried sequentially and evaluation stops at the first match. Patterns may
        include {{ground_truth}} which is resolved per-row from the ground_truth parameter.
    :type patterns: Union[str, List[str]]

    .. admonition:: Example:

        .. code-block:: python

            from azure.ai.evaluation import RegexMatchEvaluator

            # Create evaluator with a static pattern (compiled once)
            evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*B")

            result = evaluator(
                response="Based on my analysis, ANSWER: B is correct."
            )
            # result: {"regex_match": 1.0, "regex_match_result": "pass",
            #          "match_found": True, "matched_pattern_index": 0}

    .. admonition:: Example with ground_truth pattern (GPQA-style):

        .. code-block:: python

            # Pattern with ground_truth reference (resolved per-row)
            evaluator = RegexMatchEvaluator(
                patterns=r"(?i)ANSWER\s*:\s*{{ground_truth}}"
            )

            # The {{ground_truth}} is replaced with the value from ground_truth param
            result = evaluator(
                response="After analysis, ANSWER: A",
                ground_truth="A"
            )
            # result: {"regex_match": 1.0, ...}

    .. admonition:: Example with multiple patterns:

        .. code-block:: python

            evaluator = RegexMatchEvaluator(
                patterns=[
                    r"(?i)ANSWER\s*:\s*{{ground_truth}}",
                    r"(?i)The answer is\s*{{ground_truth}}",
                    r"(?i)\b{{ground_truth}}\b\s*is correct"
                ]
            )

            result = evaluator(response="I believe the answer is C.", ground_truth="C")
            # result: {"regex_match": 1.0, "matched_pattern_index": 1, ...}
    """

    id = "azureai://built-in/evaluators/regex_match"
    """Evaluator identifier, experimental and to be used only with evaluation in cloud."""

    def __init__(self, *, patterns: Union[str, List[str]]):
        """Initialize the Regex Match evaluator.

        :param patterns: One or more regex patterns to match against the response.
            Patterns are tried sequentially and evaluation stops at the first match.
            Patterns may include {{ground_truth}} which is resolved per-row from
            the ground_truth parameter.
        :type patterns: Union[str, List[str]]
        :raises ValueError: If patterns is empty or contains empty patterns.
        """
        # Normalize patterns to a list
        if isinstance(patterns, str):
            patterns = [patterns]

        if not patterns:
            raise ValueError("At least one pattern must be provided.")

        # Validate patterns are not empty
        for i, pattern in enumerate(patterns):
            if not pattern:
                raise ValueError(f"Pattern at index {i} must not be empty.")

        self._patterns = patterns

        # Check if any pattern has {{ground_truth}} reference
        self._has_dynamic_patterns = any(
            GROUND_TRUTH_PATTERN.search(p) for p in patterns
        )

        # Pre-compile static patterns (no {{ground_truth}} reference)
        if not self._has_dynamic_patterns:
            self._compiled_patterns = self._compile_patterns(patterns)
        else:
            self._compiled_patterns = None

        super().__init__()

    @staticmethod
    def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
        """Compile a list of regex patterns.

        :param patterns: List of regex pattern strings.
        :type patterns: List[str]
        :return: List of compiled regex patterns.
        :rtype: List[re.Pattern]
        :raises ValueError: If any pattern has invalid regex syntax.
        """
        compiled = []
        for i, pattern in enumerate(patterns):
            try:
                compiled.append(re.compile(pattern))
            except re.error as e:
                raise ValueError(f"Invalid regular expression pattern at index {i}: {e}")
        return compiled

    @staticmethod
    def _resolve_pattern(pattern: str, eval_input: Dict) -> str:
        """Resolve {{ground_truth}} reference in a pattern using the ground_truth value.

        Replaces {{ground_truth}} with the escaped ground_truth value from eval_input.

        :param pattern: Pattern string potentially containing {{ground_truth}} reference.
        :type pattern: str
        :param eval_input: Dictionary containing ground_truth value.
        :type eval_input: Dict
        :return: Pattern with {{ground_truth}} replaced by the actual value.
        :rtype: str
        """
        ground_truth = eval_input.get("ground_truth")
        if ground_truth is None:
            # If ground_truth is not provided, leave pattern unchanged
            return pattern

        # Escape the value for safe use in regex and replace {{ground_truth}}
        escaped_value = re.escape(str(ground_truth))
        return GROUND_TRUTH_PATTERN.sub(escaped_value, pattern)

    def _get_compiled_patterns(self, eval_input: Dict) -> List[re.Pattern]:
        """Get compiled patterns, resolving {{ground_truth}} if needed.

        For static patterns (no {{ground_truth}}), returns pre-compiled patterns.
        For dynamic patterns, resolves {{ground_truth}} and compiles per-row.

        :param eval_input: Dictionary containing ground_truth value for resolution.
        :type eval_input: Dict
        :return: List of compiled regex patterns.
        :rtype: List[re.Pattern]
        """
        if self._compiled_patterns is not None:
            # Static patterns - already compiled
            return self._compiled_patterns

        # Dynamic patterns - resolve and compile
        resolved_patterns = [
            self._resolve_pattern(p, eval_input) for p in self._patterns
        ]
        return self._compile_patterns(resolved_patterns)

    def _find_match(
        self, response: str, compiled_patterns: List[re.Pattern]
    ) -> Optional[int]:
        """Find the first matching pattern in the response.

        Tries each pattern in order and returns the index of the first match.

        :param response: The model response text to search.
        :type response: str
        :param compiled_patterns: List of compiled regex patterns to try.
        :type compiled_patterns: List[re.Pattern]
        :return: The index of the first matching pattern, or None if no match.
        :rtype: Optional[int]
        """
        if not response:
            return None

        for i, compiled_pattern in enumerate(compiled_patterns):
            if compiled_pattern.search(response):
                return i

        return None

    @override
    async def _do_eval(self, eval_input: Dict) -> Dict[str, any]:
        """Produce a regex match evaluation result.

        Checks if any pattern matches the response. For patterns with column
        references, resolves them from eval_input before matching.

        :param eval_input: The input to the evaluation function.
        :type eval_input: Dict
        :return: The evaluation result with score and match information.
        :rtype: Dict
        """
        response = eval_input.get("response", "")

        # Get compiled patterns (resolves column references if needed)
        compiled_patterns = self._get_compiled_patterns(eval_input)

        # Find first matching pattern
        matched_index = self._find_match(response, compiled_patterns)
        match_found = matched_index is not None

        # Score is 1.0 if any pattern matches, 0.0 otherwise
        score = 1.0 if match_found else 0.0

        result = {
            "regex_match": score,
            "regex_match_result": EVALUATION_PASS_FAIL_MAPPING[match_found],
            "match_found": match_found,
        }

        # Include matched pattern index for debugging if a match was found
        if match_found:
            result["matched_pattern_index"] = matched_index

        return result

    @overload  # type: ignore
    def __call__(self, *, response: str) -> Dict[str, any]:
        """
        Evaluate whether the response matches any of the configured patterns.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :return: The evaluation result containing score and match information.
        :rtype: Dict[str, any]
        """

    @overload  # type: ignore
    def __call__(self, *, response: str, ground_truth: str) -> Dict[str, any]:
        """
        Evaluate whether the response matches any of the configured patterns.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword ground_truth: The expected answer value for resolving {{ground_truth}} in patterns.
        :paramtype ground_truth: str
        :return: The evaluation result containing score and match information.
        :rtype: Dict[str, any]
        """

    @override
    def __call__(  # pylint: disable=docstring-missing-param
        self,
        *args,
        **kwargs,
    ):
        """
        Evaluate whether the response matches any of the configured patterns.

        :keyword response: The model response text to evaluate.
        :paramtype response: str
        :keyword ground_truth: The expected answer value for resolving {{ground_truth}} in patterns.
        :paramtype ground_truth: str
        :return: The evaluation result containing score and match information.
        :rtype: Dict[str, any]
        """
        return super().__call__(*args, **kwargs)
