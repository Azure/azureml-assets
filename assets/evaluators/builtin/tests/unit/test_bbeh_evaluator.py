# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

r"""Tests for the BBEHEvaluator.

These tests validate the BBEH (BIG-Bench Extra Hard) evaluator behavior:
- Answer extraction from model responses
- LaTeX formatting removal (\boxed{}, \text{}, etc.)
- Fuzzy matching logic (numeric, parentheses, quotes, etc.)
- Case-insensitive and whitespace-normalized comparison
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock the azure.ai.evaluation imports before loading the evaluator
sys.modules['azure'] = MagicMock()
sys.modules['azure.ai'] = MagicMock()
sys.modules['azure.ai.evaluation'] = MagicMock()
sys.modules['azure.ai.evaluation._evaluators'] = MagicMock()
sys.modules['azure.ai.evaluation._evaluators._common'] = MagicMock()
sys.modules['azure.ai.evaluation._constants'] = MagicMock()

# Create mock classes
mock_evaluator_base = MagicMock()
mock_evaluator_base.__class_getitem__ = MagicMock(return_value=MagicMock)
sys.modules['azure.ai.evaluation._evaluators._common'].EvaluatorBase = mock_evaluator_base
sys.modules['azure.ai.evaluation._constants'].EVALUATION_PASS_FAIL_MAPPING = {True: "pass", False: "fail"}

# Add the evaluator path to sys.path for testing
EVALUATOR_PATH = (
    Path(__file__).parent.parent.parent / "bbeh" / "evaluator"
)
sys.path.insert(0, str(EVALUATOR_PATH))

# Import only the pure functions (not the class that depends on EvaluatorBase)
from _bbeh import (  # noqa: E402
    strip_latex,
    extract_answer,
    fuzzy_match,
    evaluate_correctness,
)


# Output field names
BBEH = "bbeh"
BBEH_RESULT = "bbeh_result"

# Result values
PASS = "pass"
FAIL = "fail"


class TestStripLatex:
    """Tests for strip_latex function."""

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


class TestExtractAnswer:
    """Tests for extract_answer function."""

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


class TestFuzzyMatch:
    """Tests for fuzzy_match function."""

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


class TestEvaluateCorrectness:
    """Tests for evaluate_correctness function."""

    def test_boxed_answer(self):
        """Test extraction and matching of boxed answer."""
        assert evaluate_correctness(
            "Ok The final answer is: \\boxed{4}.", "4"
        ) is True

    def test_boxed_wrong_answer(self):
        """Test boxed answer with wrong value."""
        assert evaluate_correctness(
            "[Reasoning] The final answer is: \\boxed{4}.", "3"
        ) is False

    def test_comma_separated_list(self):
        """Test list with normalized spacing."""
        assert evaluate_correctness(
            "Alright! The final answer is: 2, 3, 4", "2,3,4"
        ) is True

    def test_comma_list_mismatch(self):
        """Test list mismatch."""
        assert evaluate_correctness(
            "blah blah The final answer is: 2, 3, 4", "2,3,5"
        ) is False

    def test_parenthesized_answer(self):
        """Test (A) matches a."""
        assert evaluate_correctness("Ok The answer is: (A)", "a") is True

    def test_parenthesized_wrong(self):
        """Test (A) doesn't match b."""
        assert evaluate_correctness("Ok The answer is: (A)", "b") is False

    def test_bold_numeric_answer(self):
        """Test bold markers removed, numeric match."""
        assert evaluate_correctness(
            "Ok The answer is: **25**\nHere's why.", "25.0"
        ) is True

    def test_bold_numeric_wrong(self):
        """Test bold markers with wrong number."""
        assert evaluate_correctness(
            "Ok The answer is: **25**\nHere's why.", "26.0"
        ) is False


class TestBBEHEvaluatorBasic:
    """Basic functionality tests for BBEH evaluation logic."""

    def test_exact_match(self):
        """Test exact string match passes."""
        result = evaluate_correctness("The answer is: yes", "yes")
        assert result is True

    def test_no_match(self):
        """Test mismatch fails."""
        result = evaluate_correctness("The answer is: no", "yes")
        assert result is False

    def test_empty_response(self):
        """Test empty response returns False."""
        result = evaluate_correctness("", "answer")
        # Empty after preprocessing - depends on fuzzy_match behavior
        assert result is False

    def test_empty_ground_truth(self):
        """Test empty ground_truth comparison."""
        result = evaluate_correctness("some response", "")
        # After preprocessing both, compare
        assert result is False


class TestBBEHEvaluatorAdvanced:
    """Advanced tests for BBEH evaluation logic."""

    def test_latex_boxed_answer(self):
        """Test LaTeX boxed answer extraction."""
        result = evaluate_correctness(
            "Let me calculate... The final answer is: \\boxed{42}.",
            "42"
        )
        assert result is True

    def test_parenthesized_mcq_answer(self):
        """Test MCQ-style parenthesized answer."""
        result = evaluate_correctness(
            "Based on the analysis, The answer is: (B)",
            "b"
        )
        assert result is True

    def test_numeric_tolerance(self):
        """Test numeric equality (float vs int)."""
        result = evaluate_correctness(
            "The answer is: 100.0",
            "100"
        )
        assert result is True

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        result = evaluate_correctness(
            "The answer is: YES",
            "yes"
        )
        assert result is True

    def test_multiline_response(self):
        """Test that only first line is considered."""
        result = evaluate_correctness(
            "The answer is: 42\nBut actually maybe it's 43",
            "42"
        )
        assert result is True
