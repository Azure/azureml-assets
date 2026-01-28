# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the RegexMatchEvaluator.

These tests validate the BabelBench GPQA-aligned regex match evaluator behavior:
- Regex is applied ONLY to the model response (not ground truth)
- Ground truth is treated as a plain expected value
- Scoring is binary: 1.0 for match, 0.0 otherwise
- Multiple regex patterns are tried sequentially
"""

import sys
from pathlib import Path
import pytest

# Add the evaluator path to sys.path for testing
EVALUATOR_PATH = (
    Path(__file__).parent.parent.parent / "regex_match" / "evaluator"
)
sys.path.insert(0, str(EVALUATOR_PATH))

from _regex_match import RegexMatchEvaluator  # noqa: E402


# Standard GPQA pattern used across tests
GPQA_PATTERN = r"(?i)ANSWER\s*:\s*([A-D])"

# Output field names
REGEX_MATCH = "regex_match"
REGEX_MATCH_RESULT = "regex_match_result"
EXTRACTED_VALUE = "extracted_value"

# Result values
PASS = "pass"
FAIL = "fail"


class TestRegexMatchEvaluatorBasic:
    """Basic functionality tests for RegexMatchEvaluator."""

    def test_gpqa_pattern_correct_match(self):
        """Test successful extraction and match with GPQA pattern."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="Based on my analysis, ANSWER: B is the correct choice.",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[REGEX_MATCH_RESULT] == PASS
        assert result[EXTRACTED_VALUE] == "B"

    def test_gpqa_pattern_incorrect_match(self):
        """Test successful extraction but wrong answer."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="I believe ANSWER: A is correct.",
            ground_truth="C"
        )
        assert result[REGEX_MATCH] == 0.0
        assert result[REGEX_MATCH_RESULT] == FAIL
        assert result[EXTRACTED_VALUE] == "A"

    def test_no_regex_match_in_response(self):
        """Test when response doesn't contain the expected pattern -> score = 0."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="I'm not sure about this question, maybe option B?",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 0.0
        assert result[REGEX_MATCH_RESULT] == FAIL
        assert result[EXTRACTED_VALUE] is None

    def test_ground_truth_is_plain_value(self):
        """Test that ground truth is NOT processed by regex - it's a plain value."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        # Ground truth is just "B", not "ANSWER: B"
        result = evaluator(
            response="ANSWER: B",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "B"

    def test_ground_truth_with_answer_prefix_fails(self):
        """Test that if ground truth has 'ANSWER: B', it won't match extracted 'B'."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="ANSWER: B",
            ground_truth="ANSWER: B"  # This is treated as literal string
        )
        # Extracted "B" != "ANSWER: B", so no match
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] == "B"


class TestRegexMatchEvaluatorCaseHandling:
    """Tests for case sensitivity handling."""

    def test_case_insensitive_match_default(self):
        """Test case-insensitive comparison is default behavior."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="answer: b",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "b"

    def test_case_insensitive_ground_truth(self):
        """Test case-insensitive comparison with lowercase ground truth."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="ANSWER: B",
            ground_truth="b"
        )
        assert result[REGEX_MATCH] == 1.0

    def test_case_sensitive_comparison(self):
        """Test case-sensitive comparison when ignore_case=False."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN, ignore_case=False)
        result = evaluator(
            response="answer: b",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] == "b"

    def test_case_sensitive_exact_match(self):
        """Test case-sensitive comparison with exact match."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN, ignore_case=False)
        result = evaluator(
            response="ANSWER: B",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 1.0


class TestRegexMatchEvaluatorMultiplePatterns:
    """Tests for multiple regex pattern support."""

    def test_first_pattern_matches(self):
        """Test that first matching pattern is used."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*([A-D])",
                r"(?i)The answer is\s*([A-D])",
            ]
        )
        result = evaluator(
            response="ANSWER: C",
            ground_truth="C"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "C"

    def test_second_pattern_matches_when_first_fails(self):
        """Test fallback to second pattern when first doesn't match."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*([A-D])",
                r"(?i)The answer is\s*([A-D])",
            ]
        )
        result = evaluator(
            response="I think the answer is D.",
            ground_truth="D"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "D"

    def test_no_pattern_matches(self):
        """Test when none of the patterns match."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*([A-D])",
                r"(?i)The answer is\s*([A-D])",
            ]
        )
        result = evaluator(
            response="I believe option B is correct.",
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] is None

    def test_single_pattern_as_string(self):
        """Test that a single pattern can be passed as string."""
        evaluator = RegexMatchEvaluator(patterns=r"Result:\s*(\d+)")
        result = evaluator(
            response="The calculation gives Result: 42",
            ground_truth="42"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "42"

    def test_three_patterns_third_matches(self):
        """Test with three patterns where third matches."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*([A-D])",
                r"(?i)The answer is\s*([A-D])",
                r"\b([A-D])\b",  # Matches standalone A-D
            ]
        )
        result = evaluator(
            response="After careful consideration, C.",
            ground_truth="C"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "C"


class TestRegexMatchEvaluatorEdgeCases:
    """Edge case tests."""

    def test_empty_response(self):
        """Test with empty response string."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(response="", ground_truth="A")
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] is None

    def test_empty_ground_truth(self):
        """Test with empty ground truth -> score = 0."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(response="ANSWER: B", ground_truth="")
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] == "B"

    def test_both_empty(self):
        """Test with both empty strings."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(response="", ground_truth="")
        assert result[REGEX_MATCH] == 0.0
        assert result[EXTRACTED_VALUE] is None

    def test_whitespace_in_answer_pattern(self):
        """Test pattern matching with various whitespace."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        # Pattern allows flexible whitespace: ANSWER\s*:\s*
        result = evaluator(
            response="ANSWER:A",  # No spaces
            ground_truth="A"
        )
        assert result[REGEX_MATCH] == 1.0

        result = evaluator(
            response="ANSWER  :   B",  # Extra spaces
            ground_truth="B"
        )
        assert result[REGEX_MATCH] == 1.0

    def test_multiple_answers_uses_first(self):
        """Test that when response contains multiple answers, first is used."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="Initially I thought ANSWER: A but actually ANSWER: B",
            ground_truth="A"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "A"

    def test_answer_at_beginning(self):
        """Test when answer appears at the beginning of response."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="ANSWER: C. This is because...",
            ground_truth="C"
        )
        assert result[REGEX_MATCH] == 1.0

    def test_answer_at_end(self):
        """Test when answer appears at the end of response."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="After analyzing all options, ANSWER: D",
            ground_truth="D"
        )
        assert result[REGEX_MATCH] == 1.0


class TestRegexMatchEvaluatorComplexResponses:
    """Tests with realistic complex model responses."""

    def test_gpqa_style_long_response(self):
        """Test with a realistic GPQA-style long response."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        response = """
        Let me analyze this question step by step.
        
        First, we need to consider the chemical properties of the compound.
        The molecular structure suggests that option A would lead to an
        unstable configuration due to steric hindrance.
        
        Option B shows the correct reaction pathway because the electron
        density distribution favors this mechanism.
        
        Options C and D are incorrect because they violate the conservation
        of angular momentum in this quantum system.
        
        Therefore, ANSWER: B
        """
        result = evaluator(response=response, ground_truth="B")
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "B"

    def test_response_with_reasoning_before_answer(self):
        """Test response with extensive reasoning before final answer."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        response = """
        The key insight here is understanding the thermodynamic equilibrium.
        When we apply Le Chatelier's principle, we can see that increasing
        pressure would shift the equilibrium to the right.
        
        Based on this analysis, ANSWER: A is the most appropriate choice.
        """
        result = evaluator(response=response, ground_truth="A")
        assert result[REGEX_MATCH] == 1.0

    def test_response_with_lowercase_answer(self):
        """Test response with lowercase 'answer:' prefix."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(
            response="Based on the evidence, answer: c",
            ground_truth="C"
        )
        assert result[REGEX_MATCH] == 1.0
        assert result[EXTRACTED_VALUE] == "c"


class TestRegexMatchEvaluatorValidation:
    """Tests for input validation."""

    def test_patterns_required(self):
        """Test that patterns parameter is required."""
        with pytest.raises(TypeError):
            RegexMatchEvaluator()

    def test_invalid_pattern_no_capture_group(self):
        """Test that pattern without capture group raises ValueError."""
        with pytest.raises(ValueError, match="at least one capture group"):
            RegexMatchEvaluator(patterns=r"ANSWER:\s*[A-D]")

    def test_invalid_pattern_syntax(self):
        """Test that invalid regex syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regular expression"):
            RegexMatchEvaluator(patterns=r"[invalid(regex")

    def test_empty_pattern_in_list(self):
        """Test that empty pattern in list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            RegexMatchEvaluator(patterns=["(?i)ANSWER:\\s*([A-D])", ""])

    def test_empty_patterns_list(self):
        """Test that empty patterns list raises ValueError."""
        with pytest.raises(ValueError, match="At least one pattern"):
            RegexMatchEvaluator(patterns=[])


class TestRegexMatchEvaluatorScoring:
    """Tests specifically for scoring behavior."""

    def test_score_is_float(self):
        """Test that score is returned as float."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(response="ANSWER: A", ground_truth="A")
        assert isinstance(result[REGEX_MATCH], float)
        assert result[REGEX_MATCH] == 1.0

    def test_score_zero_is_float(self):
        """Test that zero score is returned as float."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        result = evaluator(response="ANSWER: A", ground_truth="B")
        assert isinstance(result[REGEX_MATCH], float)
        assert result[REGEX_MATCH] == 0.0

    def test_pass_fail_mapping(self):
        """Test that pass/fail result is correctly mapped."""
        evaluator = RegexMatchEvaluator(patterns=GPQA_PATTERN)
        
        # Match -> pass
        result = evaluator(response="ANSWER: A", ground_truth="A")
        assert result[REGEX_MATCH_RESULT] == PASS
        
        # No match -> fail
        result = evaluator(response="ANSWER: A", ground_truth="B")
        assert result[REGEX_MATCH_RESULT] == FAIL
        
        # No extraction -> fail
        result = evaluator(response="I don't know", ground_truth="A")
        assert result[REGEX_MATCH_RESULT] == FAIL
