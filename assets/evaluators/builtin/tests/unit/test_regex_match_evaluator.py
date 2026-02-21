# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the RegexMatchEvaluator.

These tests validate the row-aware regex match evaluator behavior:
- Regex patterns are applied to the model response text
- Matching is existence-based (does any pattern match?)
- Scoring is binary: 1.0 if any pattern matches, 0.0 otherwise
- Multiple patterns are tried sequentially (first match wins)
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


# Output field names
REGEX_MATCH = "regex_match"
REGEX_MATCH_RESULT = "regex_match_result"
MATCH_FOUND = "match_found"
MATCHED_PATTERN_INDEX = "matched_pattern_index"

# Result values
PASS = "pass"
FAIL = "fail"


class TestRegexMatchEvaluatorBasic:
    """Basic functionality tests for RegexMatchEvaluator."""

    def test_pattern_matches_response(self):
        """Test successful pattern match in response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*B")
        result = evaluator(response="Based on my analysis, ANSWER: B is correct.")
        assert result[REGEX_MATCH] is True
        assert result[REGEX_MATCH_RESULT] == PASS
        assert result[MATCH_FOUND] is True
        assert result[MATCHED_PATTERN_INDEX] == 0

    def test_pattern_does_not_match(self):
        """Test when pattern does not match response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*B")
        result = evaluator(response="I think ANSWER: A is correct.")
        assert result[REGEX_MATCH] is False
        assert result[REGEX_MATCH_RESULT] == FAIL
        assert result[MATCH_FOUND] is False
        assert MATCHED_PATTERN_INDEX not in result

    def test_case_insensitive_flag_in_pattern(self):
        """Test case-insensitive matching using (?i) flag in pattern."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*C")
        result = evaluator(response="answer: c")
        assert result[REGEX_MATCH] is True
        assert result[MATCH_FOUND] is True

    def test_case_sensitive_pattern(self):
        """Test case-sensitive pattern matching."""
        evaluator = RegexMatchEvaluator(patterns=r"ANSWER:\s*B")
        result = evaluator(response="answer: B")
        assert result[REGEX_MATCH] is False
        assert result[MATCH_FOUND] is False

    def test_partial_match_in_response(self):
        """Test that pattern can match anywhere in response."""
        evaluator = RegexMatchEvaluator(patterns=r"correct answer is D")
        result = evaluator(
            response="After analyzing all options, the correct answer is D based on evidence."
        )
        assert result[REGEX_MATCH] is True


class TestRegexMatchEvaluatorMultiplePatterns:
    """Tests for multiple regex pattern support."""

    def test_first_pattern_matches(self):
        """Test that first matching pattern is used."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*A",
                r"(?i)The answer is\s*A",
            ]
        )
        result = evaluator(response="ANSWER: A")
        assert result[REGEX_MATCH] is True
        assert result[MATCHED_PATTERN_INDEX] == 0

    def test_second_pattern_matches(self):
        """Test fallback to second pattern when first doesn't match."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*B",
                r"(?i)The answer is\s*B",
            ]
        )
        result = evaluator(response="I think the answer is B.")
        assert result[REGEX_MATCH] is True
        assert result[MATCHED_PATTERN_INDEX] == 1

    def test_third_pattern_matches(self):
        """Test fallback to third pattern."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*C",
                r"(?i)The answer is\s*C",
                r"(?i)\bC\b\s*is correct",
            ]
        )
        result = evaluator(response="Based on the data, C is correct.")
        assert result[REGEX_MATCH] is True
        assert result[MATCHED_PATTERN_INDEX] == 2

    def test_no_pattern_matches(self):
        """Test when none of the patterns match."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*D",
                r"(?i)The answer is\s*D",
            ]
        )
        result = evaluator(response="I believe option A is correct.")
        assert result[REGEX_MATCH] is False
        assert result[MATCH_FOUND] is False

    def test_single_pattern_as_string(self):
        """Test that a single pattern can be passed as string."""
        evaluator = RegexMatchEvaluator(patterns=r"Result:\s*42")
        result = evaluator(response="The calculation gives Result: 42")
        assert result[REGEX_MATCH] is True


class TestRegexMatchEvaluatorEdgeCases:
    """Edge case tests."""

    def test_empty_response(self):
        """Test with empty response string."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*A")
        result = evaluator(response="")
        assert result[REGEX_MATCH] is False
        assert result[MATCH_FOUND] is False

    def test_whitespace_only_response(self):
        """Test with whitespace-only response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*A")
        result = evaluator(response="   \n\t   ")
        assert result[REGEX_MATCH] is False

    def test_pattern_at_beginning(self):
        """Test when pattern matches at beginning of response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*B")
        result = evaluator(response="ANSWER: B. This is because...")
        assert result[REGEX_MATCH] is True

    def test_pattern_at_end(self):
        """Test when pattern matches at end of response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*D")
        result = evaluator(response="After analyzing all options, ANSWER: D")
        assert result[REGEX_MATCH] is True

    def test_multiple_occurrences_in_response(self):
        """Test response with multiple pattern matches (first match wins)."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*[A-D]")
        result = evaluator(response="ANSWER: A but wait, ANSWER: B")
        assert result[REGEX_MATCH] is True

    def test_special_regex_characters_in_pattern(self):
        """Test pattern with special regex characters."""
        evaluator = RegexMatchEvaluator(patterns=r"Result\s*=\s*\$100\.00")
        result = evaluator(response="The Result = $100.00 after tax.")
        assert result[REGEX_MATCH] is True

    def test_unicode_in_response(self):
        """Test pattern matching with unicode characters."""
        evaluator = RegexMatchEvaluator(patterns=r"答案[:：]\s*B")
        result = evaluator(response="根据分析，答案: B")
        assert result[REGEX_MATCH] is True


class TestRegexMatchEvaluatorComplexResponses:
    """Tests with realistic complex model responses."""

    def test_gpqa_style_long_response(self):
        """Test with a realistic GPQA-style long response."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*B")
        response = (
            "Let me analyze this question step by step.\n\n"
            "First, we need to consider the chemical properties of the compound. "
            "The molecular structure suggests that option A would lead to an "
            "unstable configuration due to steric hindrance.\n\n"
            "Option B shows the correct reaction pathway because the electron "
            "density distribution favors this mechanism.\n\n"
            "Therefore, ANSWER: B"
        )
        result = evaluator(response=response)
        assert result[REGEX_MATCH] is True

    def test_response_with_wrong_answer(self):
        """Test response where model gives wrong answer."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*C")
        response = (
            "After careful analysis, I conclude that ANSWER: A is correct."
        )
        result = evaluator(response=response)
        assert result[REGEX_MATCH] is False

    def test_response_without_structured_answer(self):
        """Test response that doesn't contain the expected answer format."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*[A-D]")
        response = (
            "Based on my analysis, I think option B is the most likely answer, "
            "but I'm not entirely certain about this."
        )
        result = evaluator(response=response)
        assert result[REGEX_MATCH] is False


class TestRegexMatchEvaluatorRowAwarePatterns:
    """Tests simulating row-aware pattern behavior.

    These tests simulate how the evaluator would work when column references
    like {{item.correct_answer}} are resolved by the framework.
    """

    def test_resolved_pattern_matches(self):
        """Test pattern after column reference is resolved."""
        # Simulates: patterns="(?i)ANSWER\s*:\s*{{item.correct_answer}}"
        # where {{item.correct_answer}} = "B"
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*B")
        result = evaluator(response="ANSWER: B")
        assert result[REGEX_MATCH] is True

    def test_resolved_pattern_no_match(self):
        """Test when resolved pattern doesn't match response."""
        # Simulates correct_answer="C" but model responded with "A"
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*C")
        result = evaluator(response="ANSWER: A")
        assert result[REGEX_MATCH] is False

    def test_flexible_answer_patterns(self):
        """Test multiple patterns for flexible answer detection."""
        # Different ways a model might express the answer
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*D",
                r"(?i)The correct answer is\s*D",
                r"(?i)Option D is correct",
                r"(?i)\bD\b\s*is the answer",
            ]
        )
        result = evaluator(response="Option D is correct based on the analysis.")
        assert result[REGEX_MATCH] is True
        assert result[MATCHED_PATTERN_INDEX] == 2


class TestRegexMatchEvaluatorDynamicPatterns:
    """Tests for dynamic patterns with ground_truth reference.

    Dynamic patterns contain {{ground_truth}} reference that is resolved
    at evaluation time from the ground_truth parameter.
    """

    def test_dynamic_pattern_with_ground_truth_match(self):
        """Test dynamic pattern resolves ground_truth reference and matches."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*{{ground_truth}}")
        result = evaluator(response="ANSWER: B", ground_truth="B")
        assert result[REGEX_MATCH] is True

    def test_dynamic_pattern_with_ground_truth_no_match(self):
        """Test dynamic pattern resolves but doesn't match wrong answer."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER\s*:\s*{{ground_truth}}")
        result = evaluator(response="ANSWER: A", ground_truth="B")
        assert result[REGEX_MATCH] is False

    def test_multiple_dynamic_patterns(self):
        """Test multiple patterns with ground_truth references."""
        evaluator = RegexMatchEvaluator(
            patterns=[
                r"(?i)ANSWER\s*:\s*{{ground_truth}}",
                r"(?i)The answer is {{ground_truth}}",
            ]
        )
        result = evaluator(response="The answer is D", ground_truth="D")
        assert result[REGEX_MATCH] is True
        assert result[MATCHED_PATTERN_INDEX] == 1

    def test_dynamic_pattern_escapes_special_chars(self):
        """Test that ground_truth values with regex special chars are escaped."""
        evaluator = RegexMatchEvaluator(patterns=r"Result:\s*{{ground_truth}}")
        # The value contains regex special characters that should be escaped
        result = evaluator(response="Result: [x+y]", ground_truth="[x+y]")
        assert result[REGEX_MATCH] is True

    def test_dynamic_pattern_escapes_backslashes(self):
        r"""Test that ground_truth values with backslashes (e.g., LaTeX) are escaped.

        This tests the fix for the re.sub() backslash interpretation bug.
        When ground_truth contains backslashes like \\cos, \\theta (LaTeX),
        the escaping must work correctly.
        """
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*{{ground_truth}}")
        # LaTeX-style value with backslashes
        latex_value = r"(\cos(\theta/2), \sin(\theta/2))"
        result = evaluator(
            response=r"ANSWER: (\cos(\theta/2), \sin(\theta/2))",
            ground_truth=latex_value
        )
        assert result[REGEX_MATCH] is True

    def test_dynamic_pattern_escapes_backslash_letters(self):
        r"""Test various backslash sequences that would be invalid regex escapes.

        Tests \\c, \\p, \\l, \\h, \\m, \\i which are common in LaTeX but
        invalid as regex escape sequences.
        """
        evaluator = RegexMatchEvaluator(patterns=r"{{ground_truth}}")

        # Test \cos (contains \c)
        result = evaluator(response=r"\cos(x)", ground_truth=r"\cos(x)")
        assert result[REGEX_MATCH] is True

        # Test \pi (contains \p)
        result = evaluator(response=r"\pi", ground_truth=r"\pi")
        assert result[REGEX_MATCH] is True

        # Test \lambda (contains \l)
        result = evaluator(response=r"\lambda", ground_truth=r"\lambda")
        assert result[REGEX_MATCH] is True

        # Test \hbar (contains \h)
        result = evaluator(response=r"\hbar", ground_truth=r"\hbar")
        assert result[REGEX_MATCH] is True

    def test_mixed_static_and_dynamic_content(self):
        """Test pattern with both static regex and ground_truth reference."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)Option\s+{{ground_truth}}\s+is\s+(correct|right)")
        result = evaluator(response="Option A is correct.", ground_truth="A")
        assert result[REGEX_MATCH] is True

    def test_dynamic_pattern_case_insensitive(self):
        """Test dynamic pattern with case-insensitive flag."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i){{ground_truth}}")
        result = evaluator(response="answer: B", ground_truth="B")
        assert result[REGEX_MATCH] is True

    def test_dynamic_pattern_missing_ground_truth_no_match(self):
        """Test that missing ground_truth causes no match (pattern not resolved)."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*{{ground_truth}}")
        # When ground_truth is not provided, the pattern stays as {{ground_truth}}
        # which won't match the response, resulting in a score of 0.0
        result = evaluator(response="ANSWER: A")
        assert result[REGEX_MATCH] is False
        assert result[MATCH_FOUND] is False

    def test_dynamic_pattern_with_none_value(self):
        """Test dynamic pattern when ground_truth value is None."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*{{ground_truth}}")
        # When ground_truth is None, pattern is left unchanged and won't match
        result = evaluator(response="ANSWER: A", ground_truth=None)
        assert result[REGEX_MATCH] is False
        assert result[MATCH_FOUND] is False

    def test_dynamic_pattern_with_numeric_value(self):
        """Test dynamic pattern with numeric ground_truth value."""
        evaluator = RegexMatchEvaluator(patterns=r"Result:\s*{{ground_truth}}")
        result = evaluator(response="Result: 42", ground_truth="42")
        assert result[REGEX_MATCH] is True

    def test_static_pattern_performance_optimization(self):
        """Test that static patterns are pre-compiled (no ground_truth ref)."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*[A-D]")
        # _has_dynamic_patterns should be False for static patterns
        assert not evaluator._has_dynamic_patterns
        # _compiled_patterns should be set (not None) for static patterns
        assert evaluator._compiled_patterns is not None

    def test_dynamic_pattern_compilation_flag(self):
        """Test that dynamic patterns are detected correctly."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*{{ground_truth}}")
        # _has_dynamic_patterns should be True
        assert evaluator._has_dynamic_patterns
        # _compiled_patterns should be None (compiled per-row)
        assert evaluator._compiled_patterns is None


class TestRegexMatchEvaluatorValidation:
    """Tests for input validation."""

    def test_patterns_required(self):
        """Test that patterns parameter is required."""
        with pytest.raises(TypeError):
            RegexMatchEvaluator()

    def test_invalid_pattern_syntax(self):
        """Test that invalid regex syntax raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regular expression"):
            RegexMatchEvaluator(patterns=r"[invalid(regex")

    def test_empty_pattern_in_list(self):
        """Test that empty pattern in list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            RegexMatchEvaluator(patterns=["(?i)ANSWER:\\s*A", ""])

    def test_empty_patterns_list(self):
        """Test that empty patterns list raises ValueError."""
        with pytest.raises(ValueError, match="At least one pattern"):
            RegexMatchEvaluator(patterns=[])

    def test_pattern_without_capture_group_is_valid(self):
        """Test that patterns without capture groups are now valid."""
        # This should NOT raise - capture groups are no longer required
        evaluator = RegexMatchEvaluator(patterns=r"ANSWER:\s*[A-D]")
        result = evaluator(response="ANSWER: A")
        assert result[REGEX_MATCH] is True


class TestRegexMatchEvaluatorScoring:
    """Tests specifically for scoring behavior."""

    def test_score_is_bool_true(self):
        """Test that match score is returned as bool True."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*A")
        result = evaluator(response="ANSWER: A")
        assert isinstance(result[REGEX_MATCH], bool)
        assert result[REGEX_MATCH] is True

    def test_score_is_bool_false(self):
        """Test that no-match score is returned as bool False."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*B")
        result = evaluator(response="ANSWER: A")
        assert isinstance(result[REGEX_MATCH], bool)
        assert result[REGEX_MATCH] is False

    def test_pass_fail_mapping(self):
        """Test that pass/fail result is correctly mapped."""
        evaluator = RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*A")

        # Match -> pass
        result = evaluator(response="ANSWER: A")
        assert result[REGEX_MATCH_RESULT] == PASS

        # No match -> fail
        result = evaluator(response="ANSWER: B")
        assert result[REGEX_MATCH_RESULT] == FAIL

        # Empty response -> fail
        result = evaluator(response="")
        assert result[REGEX_MATCH_RESULT] == FAIL


class TestRegexMatchEvaluatorLogging:
    """Tests for logging behavior."""

    def test_logging_on_invalid_pattern(self, caplog):
        """Test that info logging includes pattern metadata on compilation error."""
        import logging

        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="Invalid regular expression"):
                RegexMatchEvaluator(patterns=r"[invalid(regex")

        # Check that info log includes pattern metadata
        assert any("Failed to compile pattern at index 0" in record.message for record in caplog.records)
        assert any("length=" in record.message for record in caplog.records)

    def test_logging_on_init(self, caplog):
        """Test that debug logging includes pattern count on init."""
        import logging

        with caplog.at_level(logging.DEBUG):
            RegexMatchEvaluator(patterns=[r"pattern1", r"pattern2"])

        # Check that debug log includes pattern count
        assert any("2 pattern(s)" in record.message for record in caplog.records)
        assert any("dynamic=False" in record.message for record in caplog.records)

    def test_logging_on_dynamic_pattern_init(self, caplog):
        """Test that debug logging indicates dynamic patterns."""
        import logging

        with caplog.at_level(logging.DEBUG):
            RegexMatchEvaluator(patterns=r"(?i)ANSWER:\s*{{ground_truth}}")

        # Check that dynamic flag is logged
        assert any("dynamic=True" in record.message for record in caplog.records)
