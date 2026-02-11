# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for F1 Score Evaluator."""

import pytest
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.f1_score.evaluator._f1_score import F1ScoreEvaluator


@pytest.mark.unittest
class TestF1ScoreEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for F1 Score Evaluator.

    Tests different input formats, thresholds, and edge cases.
    The F1 score evaluator normalizes text by:
    - Converting to lowercase
    - Removing punctuation
    - Removing articles (a, an, the)
    - Fixing whitespace
    Then computes precision (shared/response) and recall (shared/ground_truth).
    """

    evaluator_type = F1ScoreEvaluator
    result_key = "f1_score"

    # region Test Data
    # Perfect match scenarios
    IDENTICAL_TEXT = "The quick brown fox jumps over the lazy dog."

    # High similarity scenarios
    REFERENCE_TEXT = "The cat sat on the mat."
    SIMILAR_RESPONSE = "The cat is sitting on the mat."

    # Low similarity scenarios
    DIFFERENT_RESPONSE = "A dog runs through the park quickly."

    # Partial overlap scenarios
    PARTIAL_OVERLAP_GROUND_TRUTH = "Machine learning is a subset of artificial intelligence."
    PARTIAL_OVERLAP_RESPONSE = "Machine learning is used in artificial systems."

    # Articles test (a, an, the are removed)
    WITH_ARTICLES = "The cat sat on a mat and ate an apple."
    WITHOUT_ARTICLES = "cat sat on mat and ate apple."

    # Punctuation test
    WITH_PUNCTUATION = "Hello, world! How are you? I'm fine, thanks."
    WITHOUT_PUNCTUATION = "Hello world How are you Im fine thanks"

    # Case sensitivity test
    UPPER_CASE = "HELLO WORLD"
    LOWER_CASE = "hello world"
    MIXED_CASE = "HeLLo WoRLd"

    # Word order test (F1 is bag-of-words, order doesn't matter)
    ORIGINAL_ORDER = "apple banana cherry"
    REVERSED_ORDER = "cherry banana apple"

    # Duplicate words test
    WITH_DUPLICATES = "the cat cat cat sat on the mat mat"

    # Empty and whitespace
    EMPTY_STRING = ""
    WHITESPACE_ONLY = "   "
    SINGLE_WORD = "hello"

    # Numbers
    NUMERIC_TEXT = "The year 2024 has 365 days and 12 months."

    # Special characters
    SPECIAL_CHARS_TEXT = "Hello! How are you? I'm fine, thanks."

    # Long text
    LONG_TEXT = "This is a very long text that contains many words. " * 10

    # Subset scenarios
    SUBSET_GROUND_TRUTH = "cat mat"
    SUPERSET_RESPONSE = "the big fluffy cat sat on the soft mat"

    # No overlap
    NO_OVERLAP_TEXT1 = "apple banana cherry"
    NO_OVERLAP_TEXT2 = "dog elephant frog"

    # Single common word
    SINGLE_COMMON_1 = "hello world"
    SINGLE_COMMON_2 = "hello universe"

    # Unicode text
    UNICODE_TEXT = "café résumé naïve"
    # endregion

    # ==================== PERFECT MATCH TESTS ====================

    def test_identical_text(self):
        """Test with identical response and ground truth (should have F1 = 1.0)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Text")
        self.assert_pass(result_data)
        # Identical text should have perfect F1 score
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_identical_single_word(self):
        """Test with identical single word."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Single Word")
        self.assert_pass(result_data)
        # Identical single word should have perfect F1 score
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    # ==================== NORMALIZATION TESTS ====================

    def test_case_insensitivity(self):
        """Test that F1 score is case insensitive."""
        # Upper vs lower
        results1 = self._run_evaluation(
            response=self.UPPER_CASE,
            ground_truth=self.LOWER_CASE,
            threshold=0.5,
        )
        result_data1 = self._extract_and_print_result(results1, "Case Insensitive Upper-Lower")
        self.assert_pass(result_data1)
        self.assert_score_in_range(result_data1, min_score=1.0, max_score=1.0)

        # Mixed vs lower
        results2 = self._run_evaluation(
            response=self.MIXED_CASE,
            ground_truth=self.LOWER_CASE,
            threshold=0.5,
        )
        result_data2 = self._extract_and_print_result(results2, "Case Insensitive Mixed-Lower")
        self.assert_pass(result_data2)
        self.assert_score_in_range(result_data2, min_score=1.0, max_score=1.0)

    def test_article_removal(self):
        """Test that articles (a, an, the) are removed during normalization."""
        results = self._run_evaluation(
            response=self.WITH_ARTICLES,
            ground_truth=self.WITHOUT_ARTICLES,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Article Removal")
        # After removing articles, texts should be identical
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_punctuation_removal(self):
        """Test that punctuation is removed during normalization."""
        results = self._run_evaluation(
            response=self.WITH_PUNCTUATION,
            ground_truth=self.WITHOUT_PUNCTUATION,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Punctuation Removal")
        # After removing punctuation, texts should be identical
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_whitespace_normalization(self):
        """Test that extra whitespace is normalized."""
        response_with_spaces = "hello    world     test"
        ground_truth_normal = "hello world test"
        results = self._run_evaluation(
            response=response_with_spaces,
            ground_truth=ground_truth_normal,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Whitespace Normalization")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    # ==================== WORD ORDER TESTS (BAG OF WORDS) ====================

    def test_word_order_irrelevant(self):
        """Test that F1 score treats text as bag of words (order doesn't matter)."""
        results = self._run_evaluation(
            response=self.ORIGINAL_ORDER,
            ground_truth=self.REVERSED_ORDER,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Word Order Irrelevant")
        # Same words, different order should have F1 = 1.0
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_scrambled_sentence(self):
        """Test with scrambled sentence (same words, random order)."""
        original = "the quick brown fox jumps over the lazy dog"
        scrambled = "dog lazy the over jumps fox brown quick the"
        results = self._run_evaluation(
            response=scrambled,
            ground_truth=original,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Scrambled Sentence")
        # Same words should have F1 = 1.0 (after article removal)
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    # ==================== DUPLICATE WORDS TESTS ====================

    def test_duplicate_words_in_response(self):
        """Test handling of duplicate words - F1 considers word counts."""
        # "cat cat cat" has 3 cats, "cat" has 1 cat
        # Common = 1 cat
        # Precision = 1/3, Recall = 1/1 = 1.0
        # F1 = 2 * (1/3 * 1) / (1/3 + 1) = 2/3 / 4/3 = 0.5
        results = self._run_evaluation(
            response="cat cat cat",
            ground_truth="cat",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Duplicates in Response")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=0.5, max_score=0.5)

    def test_duplicate_words_in_ground_truth(self):
        """Test handling of duplicate words in ground truth."""
        # "cat" has 1 cat, "cat cat cat" has 3 cats
        # Common = 1 cat
        # Precision = 1/1 = 1.0, Recall = 1/3
        # F1 = 2 * (1 * 1/3) / (1 + 1/3) = 2/3 / 4/3 = 0.5
        results = self._run_evaluation(
            response="cat",
            ground_truth="cat cat cat",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Duplicates in Ground Truth")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=0.5, max_score=0.5)

    def test_matching_duplicate_counts(self):
        """Test when both have same duplicate counts."""
        results = self._run_evaluation(
            response="cat cat dog dog",
            ground_truth="cat cat dog dog",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Matching Duplicate Counts")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    # ==================== PARTIAL OVERLAP TESTS ====================

    def test_partial_overlap(self):
        """Test with partial word overlap."""
        results = self._run_evaluation(
            response=self.PARTIAL_OVERLAP_RESPONSE,
            ground_truth=self.PARTIAL_OVERLAP_GROUND_TRUTH,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Partial Overlap")
        # Some overlap, but not perfect
        self.assert_score_in_range(result_data, min_score=0.3, max_score=0.7)
        self.assert_pass(result_data)

    def test_subset_response(self):
        """Test when response is subset of ground truth."""
        results = self._run_evaluation(
            response=self.SUBSET_GROUND_TRUTH,  # "cat mat"
            ground_truth=self.SUPERSET_RESPONSE,  # longer text with cat and mat
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Subset Response")
        # High precision (all response words in ground truth), low recall
        self.assert_score_in_range(result_data, min_score=0.3, max_score=0.7)
        self.assert_pass(result_data)

    def test_superset_response(self):
        """Test when response is superset of ground truth."""
        results = self._run_evaluation(
            response=self.SUPERSET_RESPONSE,
            ground_truth=self.SUBSET_GROUND_TRUTH,  # "cat mat"
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Superset Response")
        # Low precision, high recall
        self.assert_score_in_range(result_data, min_score=0.3, max_score=0.7)
        self.assert_pass(result_data)

    # ==================== NO OVERLAP TESTS ====================

    def test_no_overlap(self):
        """Test with completely different words (no overlap)."""
        results = self._run_evaluation(
            response=self.NO_OVERLAP_TEXT1,
            ground_truth=self.NO_OVERLAP_TEXT2,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "No Overlap")
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_single_common_word(self):
        """Test with only one common word."""
        # "hello world" and "hello universe"
        # After normalization: ["hello", "world"] and ["hello", "universe"]
        # Common = 1 (hello)
        # Precision = 1/2, Recall = 1/2
        # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
        results = self._run_evaluation(
            response=self.SINGLE_COMMON_1,
            ground_truth=self.SINGLE_COMMON_2,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Single Common Word")
        self.assert_score_in_range(result_data, min_score=0.5, max_score=0.5)
        self.assert_pass(result_data)

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
            response=self.PARTIAL_OVERLAP_RESPONSE,
            ground_truth=self.PARTIAL_OVERLAP_GROUND_TRUTH,
            threshold=0.9,
        )
        result_data = self._extract_and_print_result(results, "Threshold High - Fail")
        self.assert_fail(result_data)
        assert result_data["threshold"] == 0.9

    def test_threshold_zero(self):
        """Test with zero threshold (everything should pass)."""
        results = self._run_evaluation(
            response=self.NO_OVERLAP_TEXT1,
            ground_truth=self.NO_OVERLAP_TEXT2,
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
        self.assert_pass(result_data)
        assert result_data["threshold"] == 1.0

    def test_default_threshold(self):
        """Test with default threshold (0.5)."""
        evaluator = F1ScoreEvaluator()
        results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
        result_data = self._extract_and_print_result(results, "Default Threshold")
        self.assert_pass(result_data)
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
        # Empty response means no tokens, F1 should be 0
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_empty_ground_truth(self):
        """Test with empty ground truth string."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth=self.EMPTY_STRING,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        # Empty ground truth means no tokens, F1 should be 0
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_both_empty(self):
        """Test with both empty strings."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.EMPTY_STRING,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Both Empty")
        # Both empty, no common tokens, F1 = 0
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_whitespace_only_response(self):
        """Test with whitespace-only response."""
        results = self._run_evaluation(
            response=self.WHITESPACE_ONLY,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Whitespace Only Response")
        # Whitespace only normalizes to empty, F1 = 0
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_articles_only_text(self):
        """Test with text containing only articles (a, an, the)."""
        results = self._run_evaluation(
            response="a an the a the an",
            ground_truth="hello world",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Articles Only")
        # All articles removed, no tokens left
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    def test_punctuation_only(self):
        """Test with punctuation-only text."""
        results = self._run_evaluation(
            response=".,!?;:",
            ground_truth="hello world",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Punctuation Only")
        # All punctuation removed, no tokens left
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.0)
        self.assert_fail(result_data)

    # ==================== NUMERIC TEXT TESTS ====================

    def test_numeric_text(self):
        """Test with numeric content."""
        results = self._run_evaluation(
            response=self.NUMERIC_TEXT,
            ground_truth=self.NUMERIC_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numeric Text")
        # Numbers should be preserved
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)
        self.assert_pass(result_data)

    def test_numbers_as_words(self):
        """Test numbers are treated as words."""
        results = self._run_evaluation(
            response="2024 365 12",
            ground_truth="2024 365 12",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numbers As Words")
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)
        self.assert_pass(result_data)

    def test_partial_numeric_overlap(self):
        """Test partial overlap with numbers."""
        results = self._run_evaluation(
            response="2024 is here",
            ground_truth="2024 has arrived",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Partial Numeric Overlap")
        # Only "2024" in common
        self.assert_score_in_range(result_data, min_score=0.1, max_score=0.4)
        self.assert_fail(result_data)

    # ==================== LONG TEXT TESTS ====================

    def test_long_identical_text(self):
        """Test with long identical texts."""
        results = self._run_evaluation(
            response=self.LONG_TEXT,
            ground_truth=self.LONG_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Long Identical Text")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_long_vs_short(self):
        """Test with long ground truth and short response."""
        results = self._run_evaluation(
            response="text",
            ground_truth=self.LONG_TEXT,
            threshold=0.1,
        )
        result_data = self._extract_and_print_result(results, "Long vs Short")
        # Short response will have low recall
        self.assert_score_in_range(result_data, min_score=0.0, max_score=0.1)
        self.assert_fail(result_data)

    # ==================== UNICODE TEXT TESTS ====================

    def test_unicode_text(self):
        """Test with Unicode characters."""
        results = self._run_evaluation(
            response=self.UNICODE_TEXT,
            ground_truth=self.UNICODE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Unicode Text")
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)
        self.assert_pass(result_data)

    def test_unicode_partial_match(self):
        """Test partial match with Unicode."""
        results = self._run_evaluation(
            response="café",
            ground_truth="café résumé",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Unicode Partial Match")
        self.assert_score_in_range(result_data, min_score=0.5, max_score=0.7)
        self.assert_pass(result_data)

    # ==================== PRECISION AND RECALL CALCULATION TESTS ====================

    def test_precision_recall_balance(self):
        """Test F1 balances precision and recall."""
        # Response has 2 words, ground truth has 4 words, 2 common
        # Precision = 2/2 = 1.0, Recall = 2/4 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 1.0 / 1.5 = 0.667
        results = self._run_evaluation(
            response="cat dog",
            ground_truth="cat dog bird fish",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Precision Recall Balance")
        self.assert_score_in_range(result_data, min_score=0.66, max_score=0.67)
        self.assert_pass(result_data)

    def test_high_precision_low_recall(self):
        """Test scenario with high precision but low recall."""
        # Response has all words in ground truth (precision = 1.0)
        # But ground truth has many more words (recall is low)
        results = self._run_evaluation(
            response="cat",
            ground_truth="cat dog bird fish elephant",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "High Precision Low Recall")
        # Precision = 1/1 = 1.0, Recall = 1/5 = 0.2
        # F1 = 2 * (1.0 * 0.2) / (1.0 + 0.2) = 0.4 / 1.2 = 0.333
        self.assert_score_in_range(result_data, min_score=0.33, max_score=0.34)
        self.assert_pass(result_data)

    def test_low_precision_high_recall(self):
        """Test scenario with low precision but high recall."""
        # Response has all ground truth words plus many more
        results = self._run_evaluation(
            response="cat dog bird fish elephant",
            ground_truth="cat",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Low Precision High Recall")
        # Precision = 1/5 = 0.2, Recall = 1/1 = 1.0
        # F1 = 2 * (0.2 * 1.0) / (0.2 + 1.0) = 0.4 / 1.2 = 0.333
        self.assert_score_in_range(result_data, min_score=0.33, max_score=0.34)
        self.assert_pass(result_data)

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

    # ==================== OUTPUT STRUCTURE TESTS ====================

    def test_output_contains_required_keys(self):
        """Test that output contains all required keys."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert "f1_score" in results
        assert "f1_result" in results
        assert "f1_threshold" in results

    def test_output_score_type(self):
        """Test that f1_score is a float."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert isinstance(results["f1_score"], float)

    def test_output_result_values(self):
        """Test that f1_result is either 'pass' or 'fail'."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert results["f1_result"] in ["pass", "fail"]

    def test_output_threshold_matches_input(self):
        """Test that output threshold matches input threshold."""
        threshold = 0.75
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=threshold,
        )
        assert results["f1_threshold"] == threshold

    # ==================== F1 SCORE RANGE TESTS ====================

    def test_f1_score_range_various_inputs(self):
        """Test that F1 score is always in [0, 1] range."""
        test_cases = [
            (self.IDENTICAL_TEXT, self.IDENTICAL_TEXT),
            (self.DIFFERENT_RESPONSE, self.REFERENCE_TEXT),
            (self.NO_OVERLAP_TEXT1, self.NO_OVERLAP_TEXT2),
            (self.EMPTY_STRING, self.REFERENCE_TEXT),
            (self.LONG_TEXT, self.SINGLE_WORD),
        ]
        for response, ground_truth in test_cases:
            results = self._run_evaluation(
                response=response,
                ground_truth=ground_truth,
                threshold=0.5,
            )
            if "f1_score" in results:
                assert 0.0 <= results["f1_score"] <= 1.0

    # ==================== SPECIAL ARTICLE CASES ====================

    def test_article_boundary_cases(self):
        """Test article removal doesn't affect non-article words starting with a/an/the."""
        # Words like "theater", "another", "atheist" should not be affected
        results = self._run_evaluation(
            response="theater another atheist",
            ground_truth="theater another atheist",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Article Boundary Cases")
        # These words should remain intact
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)

    def test_standalone_articles_only(self):
        """Test that only standalone articles are removed."""
        # "the" as standalone should be removed, but "there" should stay
        results = self._run_evaluation(
            response="there is the cat",
            ground_truth="there is cat",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Standalone Articles Only")
        self.assert_pass(result_data)
        self.assert_score_in_range(result_data, min_score=1.0, max_score=1.0)
