# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for BLEU Score Evaluator."""

import pytest
from typing import Any, Dict
from azure.ai.evaluation._exceptions import EvaluationException
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.bleu_score.evaluator._bleu import BleuScoreEvaluator


@pytest.mark.unittest
class TestBleuScoreEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for BLEU Score Evaluator.

    Tests different input formats, thresholds, and edge cases.
    """

    evaluator_type = BleuScoreEvaluator
    result_key = "bleu_score"

    # region Test Data
    # Perfect match scenarios
    IDENTICAL_TEXT = "The quick brown fox jumps over the lazy dog."
    
    # High similarity scenarios
    REFERENCE_TEXT = "The cat sat on the mat."
    SIMILAR_RESPONSE = "The cat is sitting on the mat."
    
    # Low similarity scenarios
    DIFFERENT_RESPONSE = "A dog runs through the park quickly."
    
    # Partial match scenarios
    PARTIAL_MATCH_REFERENCE = "Machine learning is a subset of artificial intelligence."
    PARTIAL_MATCH_RESPONSE = "Machine learning is part of AI technology."
    
    # Multi-sentence scenarios
    MULTI_SENTENCE_REFERENCE = "Hello world. This is a test. Testing is important."
    MULTI_SENTENCE_RESPONSE = "Hello world. This is a test. Testing is crucial."
    
    # Edge cases
    EMPTY_STRING = ""
    SINGLE_WORD = "Hello"
    SINGLE_CHAR = "A"
    WHITESPACE_ONLY = "   "
    PUNCTUATION_ONLY = ".,!?;:"
    NUMBERS_ONLY = "12345"
    MIXED_CASE_LOWER = "hello world"
    MIXED_CASE_UPPER = "HELLO WORLD"
    MIXED_CASE_MIXED = "Hello World"
    
    # Special characters
    SPECIAL_CHARS_TEXT = "Hello! How are you? I'm fine, thanks."
    UNICODE_TEXT = "こんにちは世界"  # Japanese "Hello World"
    UNICODE_TEXT_SIMILAR = "こんにちは"  # Japanese "Hello"
    EMOJI_TEXT = "Hello 👋 World 🌍"
    
    # Long text scenarios
    LONG_REFERENCE = "This is a very long reference text that contains many words and sentences. " * 10
    LONG_RESPONSE = "This is a very long reference text that contains many words and sentences. " * 10
    
    # Technical text
    CODE_REFERENCE = "def hello_world(): print('Hello, World!')"
    CODE_RESPONSE = "def hello_world(): print('Hello, World!')"
    
    # Numeric text
    NUMERIC_REFERENCE = "The year 2024 has 365 days and 12 months."
    NUMERIC_RESPONSE = "The year 2024 has 365 days and 12 months."
    # endregion

    # ==================== PERFECT MATCH TESTS ====================

    def test_identical_text(self):
        """Test with identical response and ground truth (should have high BLEU score)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Text")
        self.assert_pass(result_data)
        # Identical text should have perfect or near-perfect BLEU score
        assert result_data["score"] >= 0.9

    def test_identical_single_word(self):
        """Test with identical single word."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Single Word")
        self.assert_score_in_range(result_data)

    # ==================== SIMILARITY TESTS ====================

    def test_high_similarity(self):
        """Test with highly similar texts."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "High Similarity")
        self.assert_score_in_range(result_data)
        # Similar texts should have moderate to high BLEU score
        assert result_data["score"] > 0.0

    def test_low_similarity(self):
        """Test with texts having low similarity."""
        results = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Low Similarity")
        self.assert_score_in_range(result_data)
        # Different texts should have low BLEU score
        assert result_data["score"] < 0.5

    def test_partial_match(self):
        """Test with partial matching texts."""
        results = self._run_evaluation(
            response=self.PARTIAL_MATCH_RESPONSE,
            ground_truth=self.PARTIAL_MATCH_REFERENCE,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Partial Match")
        self.assert_score_in_range(result_data)

    def test_multi_sentence(self):
        """Test with multi-sentence texts."""
        results = self._run_evaluation(
            response=self.MULTI_SENTENCE_RESPONSE,
            ground_truth=self.MULTI_SENTENCE_REFERENCE,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Multi-Sentence")
        self.assert_score_in_range(result_data)
        # Most words match, should have moderate to high score
        assert result_data["score"] > 0.3

    # ==================== THRESHOLD TESTS ====================

    def test_threshold_low_pass(self):
        """Test with low threshold (should pass more easily)."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.1,
        )
        result_data = self._extract_and_print_result(results, "Threshold Low - Pass")
        self.assert_pass(result_data)
        assert result_data["threshold"] == 0.1

    def test_threshold_high_fail(self):
        """Test with high threshold (should fail more easily)."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.9,
        )
        result_data = self._extract_and_print_result(results, "Threshold High - Fail")
        self.assert_fail(result_data)
        assert result_data["threshold"] == 0.9

    def test_threshold_zero(self):
        """Test with zero threshold (everything should pass)."""
        results = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.0,
        )
        result_data = self._extract_and_print_result(results, "Threshold Zero")
        self.assert_pass(result_data)
        assert result_data["threshold"] == 0.0

    def test_threshold_one(self):
        """Test with threshold of 1.0."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=1.0,
        )
        result_data = self._extract_and_print_result(results, "Threshold One")
        # Only perfect match should pass
        self.assert_score_in_range(result_data)
        assert result_data["threshold"] == 1.0

    def test_default_threshold(self):
        """Test with default threshold (0.5)."""
        evaluator = BleuScoreEvaluator()
        results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
        result_data = self._extract_and_print_result(results, "Default Threshold")
        assert result_data["threshold"] == 0.5

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Empty Response")
        # Empty response should have zero BLEU score
        self.assert_score_in_range(result_data)
        assert result_data["score"] == 0.0

    def test_empty_ground_truth(self):
        """Test with empty ground truth string."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth=self.EMPTY_STRING,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        # Empty ground truth should result in zero BLEU score
        self.assert_score_in_range(result_data)

    def test_both_empty(self):
        """Test with both empty strings."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.EMPTY_STRING,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Both Empty")
        self.assert_score_in_range(result_data)

    def test_whitespace_only_response(self):
        """Test with whitespace-only response."""
        results = self._run_evaluation(
            response=self.WHITESPACE_ONLY,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Whitespace Only Response")
        self.assert_score_in_range(result_data)

    def test_single_character(self):
        """Test with single character texts."""
        results = self._run_evaluation(
            response=self.SINGLE_CHAR,
            ground_truth=self.SINGLE_CHAR,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Single Character")
        self.assert_score_in_range(result_data)

    def test_punctuation_only(self):
        """Test with punctuation-only text."""
        results = self._run_evaluation(
            response=self.PUNCTUATION_ONLY,
            ground_truth=self.PUNCTUATION_ONLY,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Punctuation Only")
        self.assert_score_in_range(result_data)

    def test_numbers_only(self):
        """Test with numbers-only text."""
        results = self._run_evaluation(
            response=self.NUMBERS_ONLY,
            ground_truth=self.NUMBERS_ONLY,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numbers Only")
        self.assert_score_in_range(result_data)

    # ==================== CASE SENSITIVITY TESTS ====================

    def test_case_sensitivity_lower_upper(self):
        """Test case sensitivity between lower and upper case."""
        results = self._run_evaluation(
            response=self.MIXED_CASE_LOWER,
            ground_truth=self.MIXED_CASE_UPPER,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Case Sensitivity Lower-Upper")
        self.assert_score_in_range(result_data)

    def test_case_sensitivity_lower_mixed(self):
        """Test case sensitivity between lower and mixed case."""
        results = self._run_evaluation(
            response=self.MIXED_CASE_LOWER,
            ground_truth=self.MIXED_CASE_MIXED,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Case Sensitivity Lower-Mixed")
        self.assert_score_in_range(result_data)

    # ==================== SPECIAL CHARACTERS TESTS ====================

    def test_special_characters(self):
        """Test with special characters and punctuation."""
        results = self._run_evaluation(
            response=self.SPECIAL_CHARS_TEXT,
            ground_truth=self.SPECIAL_CHARS_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Special Characters")
        self.assert_pass(result_data)

    def test_unicode_text(self):
        """Test with Unicode characters."""
        results = self._run_evaluation(
            response=self.UNICODE_TEXT,
            ground_truth=self.UNICODE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Unicode Text")
        self.assert_score_in_range(result_data)

    def test_unicode_partial_match(self):
        """Test with partial Unicode match."""
        results = self._run_evaluation(
            response=self.UNICODE_TEXT_SIMILAR,
            ground_truth=self.UNICODE_TEXT,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Unicode Partial Match")
        self.assert_score_in_range(result_data)

    def test_emoji_text(self):
        """Test with emoji characters."""
        results = self._run_evaluation(
            response=self.EMOJI_TEXT,
            ground_truth=self.EMOJI_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Emoji Text")
        self.assert_score_in_range(result_data)

    # ==================== LONG TEXT TESTS ====================

    def test_long_identical_text(self):
        """Test with long identical texts."""
        results = self._run_evaluation(
            response=self.LONG_RESPONSE,
            ground_truth=self.LONG_REFERENCE,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Long Identical Text")
        self.assert_pass(result_data)
        assert result_data["score"] >= 0.9

    def test_long_vs_short(self):
        """Test with long reference and short response."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.LONG_REFERENCE,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Long vs Short")
        self.assert_score_in_range(result_data)
        # Short response compared to long reference should have low score
        assert result_data["score"] < 0.5

    # ==================== TECHNICAL TEXT TESTS ====================

    def test_code_text(self):
        """Test with code-like text."""
        results = self._run_evaluation(
            response=self.CODE_RESPONSE,
            ground_truth=self.CODE_REFERENCE,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Code Text")
        self.assert_pass(result_data)

    def test_numeric_text(self):
        """Test with numeric text."""
        results = self._run_evaluation(
            response=self.NUMERIC_RESPONSE,
            ground_truth=self.NUMERIC_REFERENCE,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numeric Text")
        self.assert_pass(result_data)
        assert result_data["score"] >= 0.9

    # ==================== ERROR HANDLING TESTS ====================

    def test_none_response(self):
        """Test with None response."""
        results = self._run_evaluation(
            response=None,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "None Response")
        self.assert_error(result_data)

    def test_none_ground_truth(self):
        """Test with None ground truth."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth=None,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "None Ground Truth")
        self.assert_error(result_data)

    def test_both_none(self):
        """Test with both inputs as None."""
        results = self._run_evaluation(
            response=None,
            ground_truth=None,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Both None")
        self.assert_error(result_data)

    def test_invalid_response_type_int(self):
        """Test with invalid response type (int)."""
        results = self._run_evaluation(
            response=12345,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Invalid Response Type (int)")
        self.assert_error(result_data)

    def test_invalid_response_type_list(self):
        """Test with invalid response type (list)."""
        results = self._run_evaluation(
            response=["hello", "world"],
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Invalid Response Type (list)")
        self.assert_error(result_data)

    def test_invalid_ground_truth_type_dict(self):
        """Test with invalid ground truth type (dict)."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth={"text": "hello"},
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Invalid Ground Truth Type (dict)")
        self.assert_error(result_data)

    def test_negative_threshold(self):
        """Test with negative threshold - evaluator accepts it but result may always pass."""
        # The evaluator accepts negative thresholds
        # TODO: Should we enforce threshold to be within [0,1]?
        evaluator = self._init_evaluator(threshold=-0.5)
        results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
        result_data = self._extract_and_print_result(results, "Negative Threshold")
        # With negative threshold, any score >= -0.5 passes (which is always true for 0-1 scores)
        assert result_data["threshold"] == -0.5

    def test_threshold_greater_than_one(self):
        """Test with threshold greater than 1."""
        # This might be allowed or might raise an error depending on implementation
        # TODO: Should we enforce threshold to be within [0,1]?
        try:
            evaluator = self._init_evaluator(threshold=1.5)
            results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
            result_data = self._extract_and_print_result(results, "Threshold > 1")
            # If allowed, the result should be fail since max score is 1.0
            assert result_data["label"] == "fail"
        except (ValueError, EvaluationException):
            # If not allowed, exception is expected
            pass

    def test_string_threshold_type(self):
        """Test with string threshold type - evaluator may accept it."""
        # The evaluator accepts string thresholds that can be converted to float
        # TODO: Should we enforce threshold to be float type only?
        try:
            evaluator = self._init_evaluator(threshold="0.5")
            results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
            result_data = self._extract_and_print_result(results, "String Threshold")
            self.assert_score_in_range(result_data)
        except (TypeError, ValueError, EvaluationException):
            # If not allowed, exception is expected
            pass

    def test_none_threshold_type(self):
        """Test with None threshold type - evaluator uses default."""
        # The evaluator accepts None and uses default threshold
        # TODO: Should we enforce threshold to be float type only?
        try:
            evaluator = self._init_evaluator(threshold=None)
            results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
            result_data = self._extract_and_print_result(results, "None Threshold")
            self.assert_score_in_range(result_data)
        except (TypeError, ValueError, EvaluationException):
            # If not allowed, exception is expected
            pass

    # ==================== WORD ORDER TESTS ====================

    def test_same_words_different_order(self):
        """Test with same words but different order."""
        results = self._run_evaluation(
            response="dog the lazy over jumps fox brown quick the",
            ground_truth="The quick brown fox jumps over the lazy dog",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Same Words Different Order")
        self.assert_score_in_range(result_data)
        # BLEU considers n-gram order, so scrambled should have lower score
        assert result_data["score"] < 0.8

    def test_reversed_sentence(self):
        """Test with reversed sentence."""
        original = "Hello world how are you"
        reversed_text = " ".join(original.split()[::-1])
        results = self._run_evaluation(
            response=reversed_text,
            ground_truth=original,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Reversed Sentence")
        self.assert_score_in_range(result_data)

    # ==================== REPETITION TESTS ====================

    def test_repeated_words(self):
        """Test with repeated words."""
        results = self._run_evaluation(
            response="hello hello hello hello hello",
            ground_truth="hello",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Repeated Words")
        self.assert_score_in_range(result_data)

    def test_response_contains_ground_truth(self):
        """Test where response contains ground truth as substring."""
        results = self._run_evaluation(
            response="prefix " + self.REFERENCE_TEXT + " suffix",
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Response Contains Ground Truth")
        self.assert_score_in_range(result_data)
        # Should have moderate score since all ground truth words are present
        assert result_data["score"] > 0.3

    def test_ground_truth_contains_response(self):
        """Test where ground truth contains response as substring."""
        short_response = "cat sat"
        full_ground_truth = "The cat sat on the mat today"
        results = self._run_evaluation(
            response=short_response,
            ground_truth=full_ground_truth,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Ground Truth Contains Response")
        self.assert_score_in_range(result_data)

    # ==================== OUTPUT STRUCTURE TESTS ====================

    def test_output_contains_required_keys(self):
        """Test that output contains all required keys."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert "bleu_score" in results
        assert "bleu_result" in results
        assert "bleu_threshold" in results

    def test_output_score_type(self):
        """Test that bleu_score is a float."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert isinstance(results["bleu_score"], float)

    def test_output_result_values(self):
        """Test that bleu_result is either 'pass' or 'fail'."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert results["bleu_result"] in ["pass", "fail"]

    def test_output_threshold_matches_input(self):
        """Test that output threshold matches input threshold."""
        threshold = 0.75
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=threshold,
        )
        assert results["bleu_threshold"] == threshold
