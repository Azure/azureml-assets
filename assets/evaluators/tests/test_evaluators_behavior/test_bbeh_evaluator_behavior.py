# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for BBEH Evaluator."""

import pytest
from typing import Any, Dict
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.bbeh.evaluator._bbeh import (
    BBEHEvaluator,
    strip_latex,
    extract_answer,
    fuzzy_match,
)


@pytest.mark.unittest
class TestStripLatex:
    """Unit tests for the strip_latex helper."""

    def test_strip_dollar_signs(self):
        """Test stripping dollar sign wrappers."""
        assert strip_latex("$4$") == "4"
        assert strip_latex("$answer$") == "answer"

    def test_strip_boxed(self):
        r"""Test stripping \boxed{} wrapper."""
        assert strip_latex("\\boxed{4}") == "4"
        assert strip_latex("\\boxed{answer}") == "answer"

    def test_strip_text(self):
        r"""Test stripping \text{} wrapper."""
        assert strip_latex("\\text{answer}") == "answer"

    def test_strip_texttt(self):
        r"""Test stripping \texttt{} wrapper."""
        assert strip_latex("\\texttt{code}") == "code"

    def test_no_latex(self):
        """Test that plain text is unchanged."""
        assert strip_latex("plain text") == "plain text"
        assert strip_latex("42") == "42"


@pytest.mark.unittest
class TestExtractAnswer:
    """Unit tests for the extract_answer helper."""

    def test_extract_the_answer_is(self):
        """Test extracting with 'The answer is:' prefix."""
        result = extract_answer("After analysis, The answer is: 42.")
        assert result == "42"

    def test_extract_the_final_answer_is(self):
        """Test extracting with 'The final answer is' prefix."""
        result = extract_answer("Let me think... The final answer is 25")
        assert result == "25"

    def test_extract_with_latex(self):
        """Test extracting answer with LaTeX formatting."""
        result = extract_answer("The final answer is: \\boxed{4}.")
        assert result == "4"

    def test_no_answer_prefix(self):
        """Test that response without prefix returns trimmed version."""
        result = extract_answer("Just 42.")
        assert result == "Just 42"


@pytest.mark.unittest
class TestFuzzyMatch:
    """Unit tests for the fuzzy_match helper."""

    def test_exact_match(self):
        """Test exact string match."""
        assert fuzzy_match("answer", "answer") is True
        assert fuzzy_match("42", "42") is True

    def test_parenthesized_prediction(self):
        """Test (a) matches a."""
        assert fuzzy_match("(a)", "a") is True
        assert fuzzy_match("(b)", "b") is True

    def test_parenthesized_reference(self):
        """Test a matches (a)."""
        assert fuzzy_match("a", "(a)") is True
        assert fuzzy_match("b", "(b)") is True

    def test_numeric_equality(self):
        """Test numeric comparison (4.0 == 4)."""
        assert fuzzy_match("4.0", "4") is True
        assert fuzzy_match("4", "4.0") is True
        assert fuzzy_match("25.0", "25") is True

    def test_quote_normalization(self):
        """Test that single quotes are ignored."""
        assert fuzzy_match("it's", "its") is True
        assert fuzzy_match("don't", "dont") is True

    def test_bracket_variations(self):
        """Test [answer] matches answer."""
        assert fuzzy_match("[42]", "42") is True
        assert fuzzy_match("42", "[42]") is True

    def test_question_mark_ending(self):
        """Test trailing question mark is ignored."""
        assert fuzzy_match("answer?", "answer") is True

    def test_no_match(self):
        """Test that different values don't match."""
        assert fuzzy_match("yes", "no") is False
        assert fuzzy_match("42", "43") is False


@pytest.mark.unittest
class TestBBEHEvaluatorBehavior(BaseCodeEvaluatorRunner):
    r"""Behavioral tests for BBEH (BIG-Bench Extra Hard) Evaluator.

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

    # BBEH returns boolean score, not float like other evaluators
    @property
    def expected_result_fields(self):
        """Return the expected result fields for BBEH evaluator."""
        return ["bbeh", "bbeh_result"]

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

    # Quote normalization - Note: BBEH strips quotes from predictions
    QUOTE_ANSWER = "hello world"
    QUOTE_RESPONSE = "The answer is: 'hello world'"  # Single quotes are stripped

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

    # LaTeX dollar-sign wrapping ($...$)
    DOLLAR_ANSWER = "42"
    DOLLAR_RESPONSE = "The answer is: $42$"

    # LaTeX texttt formatting
    TEXTTT_ANSWER = "hello"
    TEXTTT_RESPONSE = "The final answer is: \\texttt{hello}"

    # Reference-side parenthesized MCQ (ground truth is "(b)")
    REF_PAREN_ANSWER = "(b)"
    REF_PAREN_RESPONSE = "The answer is: b"

    # Trailing question-mark ending
    QUESTION_ANSWER = "why"
    QUESTION_RESPONSE = "The answer is: why?"
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
        r"""Test LaTeX \boxed{} answer extraction."""
        results = self._run_evaluation(
            response=self.BOXED_RESPONSE,
            ground_truth=self.BOXED_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "latex_boxed_answer")
        self.assert_pass(result_data)

    def test_latex_text_answer(self):
        r"""Test LaTeX \text{} answer extraction."""
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

    # ==================== ADDITIONAL LATEX / FUZZY-MATCH BRANCH TESTS ====================

    def test_latex_dollar_sign(self):
        """Test LaTeX $...$ wrapping is stripped."""
        results = self._run_evaluation(
            response=self.DOLLAR_RESPONSE,
            ground_truth=self.DOLLAR_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "latex_dollar_sign")
        self.assert_pass(result_data)

    def test_latex_texttt_answer(self):
        r"""Test LaTeX \texttt{} answer extraction."""
        results = self._run_evaluation(
            response=self.TEXTTT_RESPONSE,
            ground_truth=self.TEXTTT_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "latex_texttt_answer")
        self.assert_pass(result_data)

    def test_reference_parenthesized_answer(self):
        """Test reference-side (b) matches bare prediction b."""
        results = self._run_evaluation(
            response=self.REF_PAREN_RESPONSE,
            ground_truth=self.REF_PAREN_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "reference_parenthesized")
        self.assert_pass(result_data)

    def test_trailing_question_mark(self):
        """Test trailing question mark is tolerated."""
        results = self._run_evaluation(
            response=self.QUESTION_RESPONSE,
            ground_truth=self.QUESTION_ANSWER,
        )
        result_data = self._extract_and_print_result(results, "trailing_question_mark")
        self.assert_pass(result_data)
