# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for BBEH Evaluator."""

import pytest
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.bbeh.evaluator._bbeh import BBEHEvaluator


@pytest.mark.unittest
class TestBBEHEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for BBEH (BIG-Bench Extra Hard) Evaluator.

    Tests the fuzzy matching logic ported from google-deepmind/bbeh:
    - LaTeX formatting (\boxed{}, \text{}, \texttt{})
    - Answer extraction from "The answer is:" prefixes
    - MCQ-style parenthesized answers
    - Numeric equality (4.0 == 4)
    - Quote normalization
    - Bracket variations
    """

    evaluator_type = BBEHEvaluator
    result_key = "bbeh"
    result_prefix = "bbeh"

    # region Test Data
    # Exact match scenarios
    SIMPLE_ANSWER = "yes"
    SIMPLE_RESPONSE = "The answer is: yes"

    # LaTeX boxed answer
    BOXED_ANSWER = "42"
    BOXED_RESPONSE = "Let me calculate... The final answer is: \\boxed{42}."

    # MCQ parenthesized answers
    MCQ_ANSWER_LOWER = "b"
    MCQ_RESPONSE_PAREN = "Based on the analysis, The answer is: (B)"

    # Numeric equality
    NUMERIC_ANSWER_INT = "100"
    NUMERIC_RESPONSE_FLOAT = "The answer is: 100.0"

    # Case insensitivity
    CASE_ANSWER_LOWER = "yes"
    CASE_RESPONSE_UPPER = "The answer is: YES"

    # Quote normalization
    QUOTE_ANSWER = "hello world"
    QUOTE_RESPONSE = "The answer is: \"hello world\""

    # Bracket variations
    BRACKET_ANSWER = "x, y, z"
    BRACKET_RESPONSE = "The answer is: [x, y, z]"

    # LaTeX text formatting
    LATEX_TEXT_ANSWER = "hello"
    LATEX_TEXT_RESPONSE = "The final answer is: \\text{hello}"

    # Multi-line (only first line should count)
    MULTILINE_ANSWER = "42"
    MULTILINE_RESPONSE = "The answer is: 42\nBut actually maybe it's 43"

    # No match scenarios
    WRONG_ANSWER = "no"
    WRONG_RESPONSE = "The answer is: yes"

    # Edge cases
    EMPTY_STRING = ""
    # endregion

    # ==================== EXACT MATCH TESTS ====================

    def test_exact_match_simple(self):
        """Test exact string match passes."""
        results = self._run_evaluation(
            response=self.SIMPLE_RESPONSE,
            ground_truth=self.SIMPLE_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "exact_match_simple")
        self.assert_pass(result_data)

    def test_exact_match_fails(self):
        """Test mismatch fails."""
        results = self._run_evaluation(
            response=self.WRONG_RESPONSE,
            ground_truth=self.WRONG_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "exact_match_fails")
        self.assert_fail(result_data)

    # ==================== LATEX FORMATTING TESTS ====================

    def test_latex_boxed_answer(self):
        """Test LaTeX \\boxed{} answer extraction."""
        results = self._run_evaluation(
            response=self.BOXED_RESPONSE,
            ground_truth=self.BOXED_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "latex_boxed_answer")
        self.assert_pass(result_data)

    def test_latex_text_answer(self):
        """Test LaTeX \\text{} answer extraction."""
        results = self._run_evaluation(
            response=self.LATEX_TEXT_RESPONSE,
            ground_truth=self.LATEX_TEXT_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "latex_text_answer")
        self.assert_pass(result_data)

    # ==================== MCQ PARENTHESIZED TESTS ====================

    def test_mcq_parenthesized_answer(self):
        """Test MCQ-style (A), (B), (C) answer matching."""
        results = self._run_evaluation(
            response=self.MCQ_RESPONSE_PAREN,
            ground_truth=self.MCQ_ANSWER_LOWER,
        )
        result_data = self._extract_and_print_result(results, "mcq_parenthesized")
        self.assert_pass(result_data)

    # ==================== NUMERIC EQUALITY TESTS ====================

    def test_numeric_equality(self):
        """Test numeric equality (4.0 == 4)."""
        results = self._run_evaluation(
            response=self.NUMERIC_RESPONSE_FLOAT,
            ground_truth=self.NUMERIC_ANSWER_INT,
        )
        result_data = self._extract_and_print_result(results, "numeric_equality")
        self.assert_pass(result_data)

    # ==================== CASE INSENSITIVITY TESTS ====================

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        results = self._run_evaluation(
            response=self.CASE_RESPONSE_UPPER,
            ground_truth=self.CASE_ANSWER_LOWER,
        )
        result_data = self._extract_and_print_result(results, "case_insensitive")
        self.assert_pass(result_data)

    # ==================== QUOTE NORMALIZATION TESTS ====================

    def test_quote_normalization(self):
        """Test quote removal from answers."""
        results = self._run_evaluation(
            response=self.QUOTE_RESPONSE,
            ground_truth=self.QUOTE_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "quote_normalization")
        self.assert_pass(result_data)

    # ==================== BRACKET VARIATION TESTS ====================

    def test_bracket_variations(self):
        """Test bracket removal [x] -> x."""
        results = self._run_evaluation(
            response=self.BRACKET_RESPONSE,
            ground_truth=self.BRACKET_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "bracket_variations")
        self.assert_pass(result_data)

    # ==================== MULTILINE TESTS ====================

    def test_multiline_response(self):
        """Test that only first line is considered."""
        results = self._run_evaluation(
            response=self.MULTILINE_RESPONSE,
            ground_truth=self.MULTILINE_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "multiline_response")
        self.assert_pass(result_data)

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test empty response returns fail."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.SIMPLE_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "empty_response")
        self.assert_fail(result_data)

    def test_empty_ground_truth(self):
        """Test empty ground_truth comparison."""
        results = self._run_evaluation(
            response=self.SIMPLE_RESPONSE,
            ground_truth=self.EMPTY_STRING,
        )
        result_data = self._extract_and_print_result(results, "empty_ground_truth")
        self.assert_fail(result_data)
