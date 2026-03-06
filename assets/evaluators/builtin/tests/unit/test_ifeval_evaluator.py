# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the IFEval evaluator instruction checkers.

These tests validate the IFEval (Instruction-Following Evaluation) behavior:
- Individual instruction checker validation
- Multiple instruction handling
- Strict vs loose accuracy modes
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock
import pytest

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
    Path(__file__).parent.parent.parent / "ifeval" / "evaluator"
)
sys.path.insert(0, str(EVALUATOR_PATH))

from _instructions import (  # noqa: E402
    CommaChecker,
    NumberOfWords,
    BulletListChecker,
    KeywordChecker,
    JsonFormat,
    PlaceholderChecker,
    TitleChecker,
    QuotationChecker,
    HighlightSectionChecker,
    get_checker,
)
from _instructions_util import count_words, count_sentences, split_into_sentences  # noqa: E402


# Output field names
IFEVAL_STRICT = "ifeval_strict"
IFEVAL_LOOSE = "ifeval_loose"
IFEVAL_RESULT = "ifeval_result"

# Result values
PASS = "pass"
FAIL = "fail"


class TestInstructionsUtil:
    """Tests for instruction utility functions."""

    def test_count_words(self):
        """Test word counting."""
        assert count_words("Hello world") == 2
        assert count_words("One two three four five") == 5
        assert count_words("") == 0
        assert count_words("Word.") == 1

    def test_count_sentences(self):
        """Test sentence counting."""
        assert count_sentences("Hello. World.") == 2
        assert count_sentences("One sentence.") == 1
        assert count_sentences("Question? Yes! Sure.") == 3

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        result = split_into_sentences("First sentence. Second sentence.")
        assert len(result) == 2
        assert "First sentence." in result[0]


class TestCommaChecker:
    """Tests for CommaChecker."""

    def test_no_commas_pass(self):
        """Test response without commas passes."""
        checker = CommaChecker()
        assert checker.check_following("This has no commas at all") is True

    def test_has_comma_fail(self):
        """Test response with comma fails."""
        checker = CommaChecker()
        assert checker.check_following("Hello, world") is False


class TestNumberOfWords:
    """Tests for NumberOfWords checker."""

    def test_at_least_pass(self):
        """Test 'at least' constraint passes."""
        checker = NumberOfWords(num_words=3, relation="at least")
        assert checker.check_following("one two three four") is True

    def test_at_least_fail(self):
        """Test 'at least' constraint fails."""
        checker = NumberOfWords(num_words=5, relation="at least")
        assert checker.check_following("one two") is False

    def test_less_than_pass(self):
        """Test 'less than' constraint passes."""
        checker = NumberOfWords(num_words=5, relation="less than")
        assert checker.check_following("one two three") is True

    def test_less_than_fail(self):
        """Test 'less than' constraint fails."""
        checker = NumberOfWords(num_words=3, relation="less than")
        assert checker.check_following("one two three four") is False

    def test_loose_tolerance(self):
        """Test loose mode allows tolerance."""
        checker = NumberOfWords(num_words=10, relation="at least")
        # Strict should fail with 8 words
        assert checker.check_following("one two three four five six seven eight") is False
        # Loose should pass (10% tolerance)
        assert checker.check_following_loose("one two three four five six seven eight") is True


class TestBulletListChecker:
    """Tests for BulletListChecker."""

    def test_asterisk_bullets_pass(self):
        """Test asterisk bullet points."""
        checker = BulletListChecker(num_bullets=3)
        response = "Here are points:\n* Point 1\n* Point 2\n* Point 3"
        assert checker.check_following(response) is True

    def test_dash_bullets_pass(self):
        """Test dash bullet points."""
        checker = BulletListChecker(num_bullets=2)
        response = "List:\n- Item 1\n- Item 2"
        assert checker.check_following(response) is True

    def test_wrong_count_fail(self):
        """Test wrong bullet count fails."""
        checker = BulletListChecker(num_bullets=5)
        response = "List:\n* Item 1\n* Item 2"
        assert checker.check_following(response) is False


class TestKeywordChecker:
    """Tests for KeywordChecker."""

    def test_keywords_present_pass(self):
        """Test all keywords present passes."""
        checker = KeywordChecker(keywords=["hello", "world"])
        assert checker.check_following("Hello world, how are you?") is True

    def test_keyword_missing_fail(self):
        """Test missing keyword fails."""
        checker = KeywordChecker(keywords=["hello", "goodbye"])
        assert checker.check_following("Hello world") is False


class TestJsonFormat:
    """Tests for JsonFormat checker."""

    def test_valid_json_pass(self):
        """Test valid JSON passes."""
        checker = JsonFormat()
        assert checker.check_following('{"key": "value"}') is True

    def test_json_with_markdown_pass(self):
        """Test JSON wrapped in markdown passes."""
        checker = JsonFormat()
        assert checker.check_following('```json\n{"key": "value"}\n```') is True

    def test_invalid_json_fail(self):
        """Test invalid JSON fails."""
        checker = JsonFormat()
        assert checker.check_following("not json at all") is False


class TestPlaceholderChecker:
    """Tests for PlaceholderChecker."""

    def test_placeholders_present_pass(self):
        """Test placeholders present passes."""
        checker = PlaceholderChecker(num_placeholders=2)
        assert checker.check_following("Dear [name], your order [order_id] is ready") is True

    def test_insufficient_placeholders_fail(self):
        """Test insufficient placeholders fails."""
        checker = PlaceholderChecker(num_placeholders=3)
        assert checker.check_following("Dear [name], hello") is False


class TestTitleChecker:
    """Tests for TitleChecker."""

    def test_title_present_pass(self):
        """Test title in angular brackets passes."""
        checker = TitleChecker()
        assert checker.check_following("<<My Title>>\nContent here") is True

    def test_no_title_fail(self):
        """Test missing title fails."""
        checker = TitleChecker()
        assert checker.check_following("No title here") is False


class TestQuotationChecker:
    """Tests for QuotationChecker."""

    def test_quoted_response_pass(self):
        """Test quoted response passes."""
        checker = QuotationChecker()
        assert checker.check_following('"This is my quoted response"') is True

    def test_unquoted_fail(self):
        """Test unquoted response fails."""
        checker = QuotationChecker()
        assert checker.check_following("This is not quoted") is False


class TestHighlightSectionChecker:
    """Tests for HighlightSectionChecker."""

    def test_highlights_present_pass(self):
        """Test highlighted sections pass."""
        checker = HighlightSectionChecker(num_highlights=2)
        assert checker.check_following("This is *important* and *critical*") is True

    def test_insufficient_highlights_fail(self):
        """Test insufficient highlights fails."""
        checker = HighlightSectionChecker(num_highlights=5)
        assert checker.check_following("Only *one* highlight") is False


class TestIFEvalEvaluatorBasic:
    """Basic functionality tests for IFEval instruction checking."""

    def test_single_instruction_pass(self):
        """Test single instruction passes."""
        checker = get_checker("punctuation:no_comma", {})
        result = checker.check_following("This response has no commas at all")
        assert result is True

    def test_single_instruction_fail(self):
        """Test single instruction fails."""
        checker = get_checker("punctuation:no_comma", {})
        result = checker.check_following("Hello, world!")
        assert result is False

    def test_unknown_instruction(self):
        """Test unknown instruction returns None."""
        checker = get_checker("unknown:instruction", {})
        assert checker is None


class TestIFEvalEvaluatorMultiple:
    """Tests for multiple instructions."""

    def test_all_instructions_pass(self):
        """Test all instructions passing."""
        response = "one two three four five six seven eight nine ten eleven twelve"
        comma_checker = get_checker("punctuation:no_comma", {})
        word_checker = get_checker("length_constraints:number_words", 
                                   {"num_words": 10, "relation": "at least"})
        assert comma_checker.check_following(response) is True
        assert word_checker.check_following(response) is True

    def test_one_instruction_fails(self):
        """Test failure when one instruction fails."""
        response = "one, two, three"  # Has commas, so no_comma fails
        comma_checker = get_checker("punctuation:no_comma", {})
        word_checker = get_checker("length_constraints:number_words",
                                   {"num_words": 3, "relation": "at least"})
        assert comma_checker.check_following(response) is False
        assert word_checker.check_following(response) is True

    def test_json_format_instruction(self):
        """Test JSON format instruction."""
        checker = get_checker("detectable_format:json_format", {})
        result = checker.check_following('{"name": "test", "value": 42}')
        assert result is True


class TestIFEvalEvaluatorLoose:
    """Tests for strict vs loose modes."""

    def test_strict_fail_loose_pass(self):
        """Test case where strict fails but loose passes."""
        # Word count requirement is 10, response has 8 words
        # Strict should fail, loose should pass (10% tolerance)
        response = "one two three four five six seven eight"
        checker = get_checker("length_constraints:number_words",
                              {"num_words": 10, "relation": "at least"})
        assert checker.check_following(response) is False
        assert checker.check_following_loose(response) is True
