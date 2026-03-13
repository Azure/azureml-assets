# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for RegexMatch Evaluator."""

import pytest
from typing import Any, Dict
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.regex_match.evaluator._regex_match import RegexMatchEvaluator


@pytest.mark.unittest
class TestRegexMatchEvaluatorBehavior(BaseCodeEvaluatorRunner):
    r"""Behavioral tests for RegexMatch Evaluator.

    Tests the regex matching capabilities:
    - Static pattern matching (compiled once at initialization)
    - Dynamic pattern matching with {{ground_truth}} resolution
    - Multiple pattern support (first match wins)
    - Case-insensitive matching via (?i) flag
    - Special character handling and escaping
    """

    evaluator_type = RegexMatchEvaluator
    result_key = "regex_match"
    result_prefix = "regex_match"
    constructor_arg_names = ["patterns"]

    # RegexMatch returns boolean score, not float
    @property
    def expected_result_fields(self):
        """Return the expected result fields for RegexMatch evaluator."""
        return ["regex_match", "regex_match_result", "match_found"]

    # Override assert methods for boolean score
    def assert_pass(self, result_data: Dict[str, Any]):
        """Assert a passing result (boolean True)."""
        assert result_data["label"] == "pass", f"Expected 'pass' but got '{result_data['label']}'"
        assert result_data["score"] is True, f"Expected True but got {result_data['score']}"

    def assert_fail(self, result_data: Dict[str, Any]):
        """Assert a failing result (boolean False)."""
        assert result_data["label"] == "fail", f"Expected 'fail' but got '{result_data['label']}'"
        assert result_data["score"] is False, f"Expected False but got {result_data['score']}"

    # region Test Data
    # Pattern matching scenarios
    SIMPLE_PATTERN = r"(?i)ANSWER\s*:\s*B"
    MATCHING_RESPONSE = "Based on my analysis, ANSWER: B is correct."
    NON_MATCHING_RESPONSE = "I think ANSWER: A is correct."

    # Case sensitivity
    CASE_SENSITIVE_PATTERN = r"ANSWER:\s*B"
    LOWERCASE_RESPONSE = "answer: B"

    # Multiple patterns
    MULTI_PATTERNS = [
        r"(?i)ANSWER\s*:\s*B",
        r"(?i)The answer is\s*B",
    ]
    SECOND_PATTERN_RESPONSE = "I think the answer is B."

    # Dynamic pattern with ground_truth
    DYNAMIC_PATTERN = r"(?i)ANSWER\s*:\s*{{ground_truth}}"
    DYNAMIC_RESPONSE = "After analysis, ANSWER: A"
    DYNAMIC_GROUND_TRUTH = "A"
    DYNAMIC_WRONG_GROUND_TRUTH = "B"

    # Special characters
    SPECIAL_CHAR_PATTERN = r"Result\s*=\s*\$100\.00"
    SPECIAL_CHAR_RESPONSE = "The Result = $100.00 after tax."

    # Edge cases
    EMPTY_STRING = ""
    WHITESPACE_RESPONSE = "   \n\t   "
    # endregion

    # ==================== BASIC PATTERN MATCHING TESTS ====================

    def test_pattern_matches_response(self):
        """Test successful pattern match in response."""
        evaluator = RegexMatchEvaluator(patterns=self.SIMPLE_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.MATCHING_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "pattern_matches")
        self.assert_pass(result_data)

    def test_pattern_does_not_match(self):
        """Test when pattern does not match response."""
        evaluator = RegexMatchEvaluator(patterns=self.SIMPLE_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.NON_MATCHING_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "pattern_no_match")
        self.assert_fail(result_data)

    def test_case_sensitive_pattern(self):
        """Test case-sensitive pattern matching."""
        evaluator = RegexMatchEvaluator(patterns=self.CASE_SENSITIVE_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.LOWERCASE_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "case_sensitive")
        self.assert_fail(result_data)

    # ==================== MULTIPLE PATTERNS TESTS ====================

    def test_second_pattern_matches(self):
        """Test fallback to second pattern when first doesn't match."""
        evaluator = RegexMatchEvaluator(patterns=self.MULTI_PATTERNS)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.SECOND_PATTERN_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "second_pattern")
        self.assert_pass(result_data)
        # Verify pattern index in raw results
        assert results.get("matched_pattern_index") == 1

    def test_first_pattern_matches(self):
        """Test that first matching pattern is used."""
        evaluator = RegexMatchEvaluator(patterns=self.MULTI_PATTERNS)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.MATCHING_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "first_pattern")
        self.assert_pass(result_data)
        assert results.get("matched_pattern_index") == 0

    # ==================== DYNAMIC PATTERN TESTS ====================

    def test_dynamic_pattern_with_ground_truth_match(self):
        """Test dynamic pattern resolution with matching ground_truth."""
        evaluator = RegexMatchEvaluator(patterns=self.DYNAMIC_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.DYNAMIC_RESPONSE,
            ground_truth=self.DYNAMIC_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "dynamic_match")
        self.assert_pass(result_data)

    def test_dynamic_pattern_with_ground_truth_no_match(self):
        """Test dynamic pattern resolution with non-matching ground_truth."""
        evaluator = RegexMatchEvaluator(patterns=self.DYNAMIC_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.DYNAMIC_RESPONSE,
            ground_truth=self.DYNAMIC_WRONG_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "dynamic_no_match")
        self.assert_fail(result_data)

    # ==================== SPECIAL CHARACTERS TESTS ====================

    def test_special_regex_characters_in_pattern(self):
        """Test pattern with special regex characters."""
        evaluator = RegexMatchEvaluator(patterns=self.SPECIAL_CHAR_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.SPECIAL_CHAR_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "special_chars")
        self.assert_pass(result_data)

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string."""
        evaluator = RegexMatchEvaluator(patterns=self.SIMPLE_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.EMPTY_STRING,
        )
        result_data = self._extract_and_print_result(results, "empty_response")
        self.assert_fail(result_data)

    def test_whitespace_only_response(self):
        """Test with whitespace-only response."""
        evaluator = RegexMatchEvaluator(patterns=self.SIMPLE_PATTERN)
        results = self._run_evaluation_with_evaluator(
            evaluator,
            response=self.WHITESPACE_RESPONSE,
        )
        result_data = self._extract_and_print_result(results, "whitespace_response")
        self.assert_fail(result_data)

    # ==================== HELPER METHOD ====================

    def _run_evaluation_with_evaluator(self, evaluator, **kwargs):
        """Run evaluation with a pre-constructed evaluator."""
        return evaluator(**kwargs)
