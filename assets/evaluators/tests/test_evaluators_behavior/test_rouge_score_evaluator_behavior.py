# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for ROUGE Score Evaluator."""

import pytest
import math
from typing import Any, Dict, List, override
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.rouge_score.evaluator._rouge import RougeScoreEvaluator, RougeType


@pytest.mark.unittest
class TestRougeScoreEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for ROUGE Score Evaluator.

    Tests different ROUGE types, thresholds, and edge cases.
    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) evaluates n-gram overlap:
    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-3: Trigram overlap
    - ROUGE-4: 4-gram overlap
    - ROUGE-5: 5-gram overlap
    - ROUGE-L: Longest common subsequence

    Returns three metrics: precision, recall, and F1 score (each with separate thresholds).
    Scores range from 0 to 1, with higher being better.
    """

    evaluator_type = RougeScoreEvaluator
    result_key = "rouge_f1_score"
    result_prefix = "rouge"
    constructor_arg_names = ["rouge_type", "precision_threshold", "recall_threshold", "f1_score_threshold"]

    @property
    def expected_result_fields(self) -> List[str]:
        return [
            f"{self.result_prefix}_precision",
            f"{self.result_prefix}_recall",
            f"{self.result_prefix}_f1_score",
            f"{self.result_prefix}_precision_result",
            f"{self.result_prefix}_recall_result",
            f"{self.result_prefix}_f1_score_result",
            f"{self.result_prefix}_precision_threshold",
            f"{self.result_prefix}_recall_threshold",
            f"{self.result_prefix}_f1_score_threshold",
        ]

    # region Test Data
    # Perfect match scenarios
    IDENTICAL_TEXT = "The quick brown fox jumps over the lazy dog."

    # High similarity scenarios
    REFERENCE_TEXT = "The cat sat on the mat."
    SIMILAR_RESPONSE = "The cat is sitting on the mat."

    # Low similarity scenarios
    DIFFERENT_RESPONSE = "A dog runs through the park quickly."

    # Partial match scenarios
    PARTIAL_MATCH_GROUND_TRUTH = "Machine learning is a subset of artificial intelligence."
    PARTIAL_MATCH_RESPONSE = "Machine learning is part of AI technology."

    # Multi-sentence scenarios
    MULTI_SENTENCE_GROUND_TRUTH = "Hello world. This is a test. Testing is important."
    MULTI_SENTENCE_RESPONSE = "Hello world. This is a test. Testing is crucial."

    # N-gram specific test data
    # For ROUGE-2 (bigram) testing
    BIGRAM_GROUND_TRUTH = "the cat sat on the mat"
    BIGRAM_PERFECT_RESPONSE = "the cat sat on the mat"
    BIGRAM_PARTIAL_RESPONSE = "the cat ran on the floor"  # "the cat" and "on the" match
    BIGRAM_NO_MATCH_RESPONSE = "a dog walks in a park"  # No bigram matches

    # For ROUGE-L (longest common subsequence) testing
    LCS_GROUND_TRUTH = "police killed the gunman"
    LCS_RESPONSE = "police kill the gunman"  # LCS = "police the gunman"

    # Word order matters for n-grams
    ORDERED_TEXT = "apple banana cherry date elderberry"
    REVERSED_TEXT = "elderberry date cherry banana apple"

    # Edge cases
    EMPTY_STRING = ""
    SINGLE_WORD = "Hello"
    SINGLE_CHAR = "A"
    WHITESPACE_ONLY = "   "
    PUNCTUATION_ONLY = ".,!?;:"
    NUMBERS_ONLY = "12345"

    # Short texts for higher n-gram tests
    SHORT_TEXT = "cat sat"  # Only 2 words - no trigrams possible
    THREE_WORD_TEXT = "the cat sat"  # 3 words - exactly 1 trigram

    # Case variations
    MIXED_CASE_LOWER = "hello world test"
    MIXED_CASE_UPPER = "HELLO WORLD TEST"
    MIXED_CASE_MIXED = "Hello World Test"

    # Special characters
    SPECIAL_CHARS_TEXT = "Hello! How are you? I'm fine, thanks."
    UNICODE_TEXT = "café résumé naïve"

    # Long text scenarios
    LONG_TEXT = "This is a very long text that contains many words and sentences. " * 10

    # Precision vs Recall test data
    # Short response (high precision, low recall)
    HIGH_PRECISION_RESPONSE = "cat sat"
    HIGH_PRECISION_GROUND_TRUTH = "the big fluffy cat sat on the soft mat"

    # Long response (low precision, high recall)
    LOW_PRECISION_RESPONSE = "the big fluffy cat sat on the soft mat with many extra words added"
    LOW_PRECISION_GROUND_TRUTH = "cat sat mat"
    # endregion

    @override
    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Extract result fields specific for Rouge Score Evaluator and print them.

        Args:
            results: Raw evaluation results from the evaluator.
            test_label: Label for the test (used in print output).

        Returns:
            Dictionary with standardized result fields.
        """
        if f"{self.result_key}_error_message" not in results:
            for field in self.expected_result_fields:
                if field not in results:
                    raise ValueError(f"Expected result field '{field}' not found in results.")

        precision = results.get("rouge_precision")
        recall = results.get("rouge_recall")
        f1_score = results.get("rouge_f1_score")
        precision_result = results.get("rouge_precision_result")
        recall_result = results.get("rouge_recall_result")
        f1_result = results.get("rouge_f1_score_result")
        precision_threshold = results.get("rouge_precision_threshold")
        recall_threshold = results.get("rouge_recall_threshold")
        f1_threshold = results.get("rouge_f1_score_threshold")
        error_message = results.get("rouge_f1_score_error_message")
        error_code = results.get("rouge_f1_score_error_code")

        print(f"\n[{test_label}]")
        print(f"  Precision: {precision} (result: {precision_result}, threshold: {precision_threshold})")
        print(f"  Recall: {recall} (result: {recall_result}, threshold: {recall_threshold})")
        print(f"  F1 Score: {f1_score} (result: {f1_result}, threshold: {f1_threshold})")
        if error_message or error_code:
            print(f"  Error Message: {error_message}")
            print(f"  Error Code: {error_code}")

        return {
            "evaluator_name": "rouge",
            "score": f1_score,
            "label": f1_result,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "precision_result": precision_result,
            "recall_result": recall_result,
            "f1_result": f1_result,
            "precision_threshold": precision_threshold,
            "recall_threshold": recall_threshold,
            "f1_threshold": f1_threshold,
            "error_message": error_message,
            "error_code": error_code,
        }

    def assert_all_pass(self, result_data: Dict[str, Any]):
        """Assert all metrics pass."""
        assert result_data["precision_result"] == "pass"
        assert result_data["recall_result"] == "pass"
        assert result_data["f1_result"] == "pass"

    def assert_all_fail(self, result_data: Dict[str, Any]):
        """Assert all metrics fail."""
        assert result_data["precision_result"] == "fail"
        assert result_data["recall_result"] == "fail"
        assert result_data["f1_result"] == "fail"

    def assert_scores_in_range(self, result_data: Dict[str, Any], min_score: float = 0.0, max_score: float = 1.0):
        """Assert that all scores are within expected range."""
        if result_data["precision"] is not None and not math.isnan(result_data["precision"]):
            assert min_score <= result_data["precision"] <= max_score
        if result_data["recall"] is not None and not math.isnan(result_data["recall"]):
            assert min_score <= result_data["recall"] <= max_score
        if result_data["f1_score"] is not None and not math.isnan(result_data["f1_score"]):
            assert min_score <= result_data["f1_score"] <= max_score

    # ==================== PERFECT MATCH TESTS ====================

    def test_identical_text_rouge1(self):
        """Test with identical text using ROUGE-1 (unigram)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Identical Text ROUGE-1")
        self.assert_all_pass(result_data)
        # Identical text should have perfect scores
        assert result_data["precision"] == 1.0
        assert result_data["recall"] == 1.0
        assert result_data["f1_score"] == 1.0

    def test_identical_text_rouge2(self):
        """Test with identical text using ROUGE-2 (bigram)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_2,
        )
        result_data = self._extract_and_print_result(results, "Identical Text ROUGE-2")
        self.assert_all_pass(result_data)
        assert result_data["precision"] == 1.0
        assert result_data["recall"] == 1.0
        assert result_data["f1_score"] == 1.0

    def test_identical_text_rouge_l(self):
        """Test with identical text using ROUGE-L (LCS)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_L,
        )
        result_data = self._extract_and_print_result(results, "Identical Text ROUGE-L")
        self.assert_all_pass(result_data)
        assert result_data["precision"] == 1.0
        assert result_data["recall"] == 1.0
        assert result_data["f1_score"] == 1.0

    # ==================== ROUGE TYPE TESTS ====================

    def test_rouge1_unigram_matching(self):
        """Test ROUGE-1 unigram matching."""
        results = self._run_evaluation(
            response=self.BIGRAM_PARTIAL_RESPONSE,
            ground_truth=self.BIGRAM_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=0.3,
            recall_threshold=0.3,
            f1_score_threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-1 Unigram")
        self.assert_scores_in_range(result_data)
        # Some unigram overlap expected
        assert result_data["f1_score"] > 0.0

    def test_rouge2_bigram_matching(self):
        """Test ROUGE-2 bigram matching."""
        results = self._run_evaluation(
            response=self.BIGRAM_PARTIAL_RESPONSE,
            ground_truth=self.BIGRAM_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_2,
            precision_threshold=0.1,
            recall_threshold=0.1,
            f1_score_threshold=0.1,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-2 Bigram")
        self.assert_scores_in_range(result_data)

    def test_rouge2_no_bigram_match(self):
        """Test ROUGE-2 with no bigram matches."""
        results = self._run_evaluation(
            response=self.BIGRAM_NO_MATCH_RESPONSE,
            ground_truth=self.BIGRAM_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_2,
            precision_threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-2 No Match")
        self.assert_scores_in_range(result_data)
        # No bigram overlap
        assert result_data["f1_score"] == 0.0

    def test_rouge3_trigram_matching(self):
        """Test ROUGE-3 trigram matching."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_3,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-3 Trigram")
        self.assert_all_pass(result_data)
        assert result_data["f1_score"] == 1.0

    def test_rouge4_fourgram_matching(self):
        """Test ROUGE-4 four-gram matching."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_4,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-4 Four-gram")
        self.assert_all_pass(result_data)

    def test_rouge5_fivegram_matching(self):
        """Test ROUGE-5 five-gram matching."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_5,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-5 Five-gram")
        self.assert_all_pass(result_data)

    def test_rouge_l_lcs_matching(self):
        """Test ROUGE-L longest common subsequence matching."""
        results = self._run_evaluation(
            response=self.LCS_RESPONSE,
            ground_truth=self.LCS_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_L,
            precision_threshold=0.3,
            recall_threshold=0.3,
            f1_score_threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "ROUGE-L LCS")
        self.assert_scores_in_range(result_data)
        # Should have reasonable LCS score
        assert result_data["f1_score"] > 0.5

    def test_higher_ngram_lower_scores(self):
        """Test that higher n-grams produce lower scores for partial matches."""
        results_rouge1 = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        results_rouge2 = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_2,
        )

        result1 = self._extract_and_print_result(results_rouge1, "Higher N-gram ROUGE-1")
        result2 = self._extract_and_print_result(results_rouge2, "Higher N-gram ROUGE-2")

        # ROUGE-1 should generally have higher or equal score than ROUGE-2 for partial matches
        assert result1["f1_score"] >= result2["f1_score"]

    # ==================== PRECISION VS RECALL TESTS ====================

    def test_high_precision_low_recall(self):
        """Test scenario with high precision but low recall."""
        # Short response that's all correct = high precision
        # But missing most of ground truth = low recall
        results = self._run_evaluation(
            response=self.HIGH_PRECISION_RESPONSE,
            ground_truth=self.HIGH_PRECISION_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=0.5,
            recall_threshold=0.1,
            f1_score_threshold=0.2,
        )
        result_data = self._extract_and_print_result(results, "High Precision Low Recall")
        self.assert_scores_in_range(result_data)
        # Precision should be higher than recall
        assert result_data["precision"] > result_data["recall"]

    def test_low_precision_high_recall(self):
        """Test scenario with low precision but high recall."""
        # Long response covering all ground truth = high recall
        # But with extra words = low precision
        results = self._run_evaluation(
            response=self.LOW_PRECISION_RESPONSE,
            ground_truth=self.LOW_PRECISION_GROUND_TRUTH,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=0.1,
            recall_threshold=0.5,
            f1_score_threshold=0.2,
        )
        result_data = self._extract_and_print_result(results, "Low Precision High Recall")
        self.assert_scores_in_range(result_data)
        # Recall should be higher than precision
        assert result_data["recall"] > result_data["precision"]

    # ==================== THRESHOLD TESTS ====================

    def test_separate_thresholds(self):
        """Test that each metric has its own threshold."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=0.1,  # Low - should pass
            recall_threshold=0.9,      # High - should fail
            f1_score_threshold=0.5,    # Medium
        )
        result_data = self._extract_and_print_result(results, "Separate Thresholds")

        # Check thresholds are stored correctly
        assert result_data["precision_threshold"] == 0.1
        assert result_data["recall_threshold"] == 0.9
        assert result_data["f1_threshold"] == 0.5

        # Precision should pass with low threshold
        assert result_data["precision_result"] == "pass"
        # Recall should fail with high threshold
        assert result_data["recall_result"] == "fail"

    def test_all_thresholds_zero(self):
        """Test with all thresholds at zero (everything passes)."""
        results = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=0.0,
            recall_threshold=0.0,
            f1_score_threshold=0.0,
        )
        result_data = self._extract_and_print_result(results, "All Thresholds Zero")
        self.assert_all_pass(result_data)

    def test_all_thresholds_one(self):
        """Test with all thresholds at 1.0."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_1,
            precision_threshold=1.0,
            recall_threshold=1.0,
            f1_score_threshold=1.0,
        )
        result_data = self._extract_and_print_result(results, "All Thresholds One")
        # Identical text should still pass
        self.assert_all_pass(result_data)

    def test_default_thresholds(self):
        """Test with default thresholds (0.5 for all)."""
        evaluator = RougeScoreEvaluator(rouge_type=RougeType.ROUGE_1)
        results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)

        assert results["rouge_precision_threshold"] == 0.5
        assert results["rouge_recall_threshold"] == 0.5
        assert results["rouge_f1_score_threshold"] == 0.5

    # ==================== WORD ORDER TESTS ====================

    def test_word_order_affects_ngrams(self):
        """Test that word order affects n-gram scores."""
        # Same words, different order
        results_ordered = self._run_evaluation(
            response=self.ORDERED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            rouge_type=RougeType.ROUGE_2,
        )
        results_reversed = self._run_evaluation(
            response=self.REVERSED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            rouge_type=RougeType.ROUGE_2,
        )

        result_ordered = self._extract_and_print_result(results_ordered, "Word Order - Ordered")
        result_reversed = self._extract_and_print_result(results_reversed, "Word Order - Reversed")

        # Ordered should have perfect score
        assert result_ordered["f1_score"] == 1.0
        # Reversed should have lower score for bigrams
        assert result_reversed["f1_score"] < result_ordered["f1_score"]

    def test_rouge1_less_affected_by_order(self):
        """Test that ROUGE-1 is less affected by word order than ROUGE-2."""
        results_rouge1 = self._run_evaluation(
            response=self.REVERSED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        results_rouge2 = self._run_evaluation(
            response=self.REVERSED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            rouge_type=RougeType.ROUGE_2,
        )

        result1 = self._extract_and_print_result(results_rouge1, "Order Effect ROUGE-1")
        result2 = self._extract_and_print_result(results_rouge2, "Order Effect ROUGE-2")

        # ROUGE-1 (unigrams) should still be perfect for reversed words
        assert result1["f1_score"] == 1.0
        # ROUGE-2 (bigrams) should be much lower
        assert result2["f1_score"] < result1["f1_score"]

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Empty Response")
        # Empty response should have zero scores
        assert result_data["precision"] == 0.0 or math.isnan(result_data["precision"])
        assert result_data["recall"] == 0.0

    def test_empty_ground_truth(self):
        """Test with empty ground truth string."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth=self.EMPTY_STRING,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        self.assert_scores_in_range(result_data)

    def test_both_empty(self):
        """Test with both empty strings."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.EMPTY_STRING,
            rouge_type=RougeType.ROUGE_1,
        )
        self._extract_and_print_result(results, "Both Empty")
        # Both empty - scores may be 0 or NaN

    def test_single_word(self):
        """Test with single word texts."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Single Word")
        assert result_data["f1_score"] == 1.0

    def test_single_word_rouge2(self):
        """Test ROUGE-2 with single word (no bigrams possible)."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
            rouge_type=RougeType.ROUGE_2,
        )
        self._extract_and_print_result(results, "Single Word ROUGE-2")
        # No bigrams in single word

    def test_short_text_higher_ngrams(self):
        """Test higher n-grams with text too short to have those n-grams."""
        # "cat sat" has only 2 words - no trigrams possible
        results = self._run_evaluation(
            response=self.SHORT_TEXT,
            ground_truth=self.SHORT_TEXT,
            rouge_type=RougeType.ROUGE_3,
        )
        self._extract_and_print_result(results, "Short Text ROUGE-3")
        # No trigrams possible in 2-word text

    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        results = self._run_evaluation(
            response=self.WHITESPACE_ONLY,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        self._extract_and_print_result(results, "Whitespace Only")

    def test_punctuation_only(self):
        """Test with punctuation-only text."""
        results = self._run_evaluation(
            response=self.PUNCTUATION_ONLY,
            ground_truth=self.PUNCTUATION_ONLY,
            rouge_type=RougeType.ROUGE_1,
        )
        self._extract_and_print_result(results, "Punctuation Only")

    # ==================== CASE SENSITIVITY TESTS ====================

    def test_case_handling(self):
        """Test case handling between lower and upper case."""
        results = self._run_evaluation(
            response=self.MIXED_CASE_LOWER,
            ground_truth=self.MIXED_CASE_UPPER,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Case Handling")
        self.assert_scores_in_range(result_data)

    # ==================== SPECIAL CHARACTERS TESTS ====================

    def test_special_characters(self):
        """Test with special characters and punctuation."""
        results = self._run_evaluation(
            response=self.SPECIAL_CHARS_TEXT,
            ground_truth=self.SPECIAL_CHARS_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Special Characters")
        self.assert_all_pass(result_data)

    def test_unicode_text(self):
        """Test with Unicode characters."""
        results = self._run_evaluation(
            response=self.UNICODE_TEXT,
            ground_truth=self.UNICODE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Unicode Text")
        self.assert_scores_in_range(result_data)

    # ==================== LONG TEXT TESTS ====================

    def test_long_identical_text(self):
        """Test with long identical texts."""
        results = self._run_evaluation(
            response=self.LONG_TEXT,
            ground_truth=self.LONG_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Long Identical Text")
        self.assert_all_pass(result_data)
        assert result_data["f1_score"] == 1.0

    def test_long_vs_short(self):
        """Test with long ground truth and short response."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.LONG_TEXT,
            rouge_type=RougeType.ROUGE_1,
            recall_threshold=0.1,
        )
        result_data = self._extract_and_print_result(results, "Long vs Short")
        self.assert_scores_in_range(result_data)

    # ==================== ERROR HANDLING TESTS ====================

    def test_none_response(self):
        """Test with None response."""
        results = self._run_evaluation(
            response=None,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "None Response")
        self.assert_error(result_data)

    def test_none_ground_truth(self):
        """Test with None ground truth."""
        results = self._run_evaluation(
            response=self.REFERENCE_TEXT,
            ground_truth=None,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "None Ground Truth")
        self.assert_error(result_data)

    def test_invalid_response_type(self):
        """Test with invalid response type (int)."""
        results = self._run_evaluation(
            response=12345,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "Invalid Response Type")
        self.assert_error(result_data)

    def test_invalid_threshold_type_int(self):
        """Test with invalid threshold type (int instead of float)."""
        with pytest.raises(TypeError):
            self._init_evaluator(
                rouge_type=RougeType.ROUGE_1,
                precision_threshold=1,  # int instead of float
            )

    def test_invalid_threshold_type_string(self):
        """Test with invalid threshold type (string)."""
        with pytest.raises(TypeError):
            self._init_evaluator(
                rouge_type=RougeType.ROUGE_1,
                recall_threshold="0.5",  # string instead of float
            )

    # ==================== ROUGE TYPE ENUM TESTS ====================

    def test_all_rouge_types_work(self):
        """Test that all RougeType enum values work."""
        rouge_types = [
            RougeType.ROUGE_1,
            RougeType.ROUGE_2,
            RougeType.ROUGE_3,
            RougeType.ROUGE_4,
            RougeType.ROUGE_5,
            RougeType.ROUGE_L,
        ]
        for rouge_type in rouge_types:
            results = self._run_evaluation(
                response=self.IDENTICAL_TEXT,
                ground_truth=self.IDENTICAL_TEXT,
                rouge_type=rouge_type,
            )
            result_data = self._extract_and_print_result(results, f"RougeType {rouge_type.value}")
            self.assert_scores_in_range(result_data)

    def test_rouge_type_string_value(self):
        """Test RougeType enum string values."""
        assert RougeType.ROUGE_1.value == "rouge1"
        assert RougeType.ROUGE_2.value == "rouge2"
        assert RougeType.ROUGE_3.value == "rouge3"
        assert RougeType.ROUGE_4.value == "rouge4"
        assert RougeType.ROUGE_5.value == "rouge5"
        assert RougeType.ROUGE_L.value == "rougeL"

    # ==================== OUTPUT STRUCTURE TESTS ====================

    def test_output_contains_all_keys(self):
        """Test that output contains all required keys."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        # Score keys
        assert "rouge_precision" in results
        assert "rouge_recall" in results
        assert "rouge_f1_score" in results
        # Result keys
        assert "rouge_precision_result" in results
        assert "rouge_recall_result" in results
        assert "rouge_f1_score_result" in results
        # Threshold keys
        assert "rouge_precision_threshold" in results
        assert "rouge_recall_threshold" in results
        assert "rouge_f1_score_threshold" in results

    def test_output_score_types(self):
        """Test that scores are floats."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        assert isinstance(results["rouge_precision"], float)
        assert isinstance(results["rouge_recall"], float)
        assert isinstance(results["rouge_f1_score"], float)

    def test_output_result_values(self):
        """Test that results are 'pass' or 'fail'."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        assert results["rouge_precision_result"] in ["pass", "fail"]
        assert results["rouge_recall_result"] in ["pass", "fail"]
        assert results["rouge_f1_score_result"] in ["pass", "fail"]

    # ==================== F1 SCORE CALCULATION TESTS ====================

    def test_f1_is_harmonic_mean(self):
        """Test that F1 score is harmonic mean of precision and recall."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        result_data = self._extract_and_print_result(results, "F1 Harmonic Mean")

        p = result_data["precision"]
        r = result_data["recall"]
        f1 = result_data["f1_score"]

        if p > 0 and r > 0:
            expected_f1 = 2 * (p * r) / (p + r)
            assert abs(f1 - expected_f1) < 0.0001

    # ==================== SIMILARITY COMPARISON TESTS ====================

    def test_similar_vs_different(self):
        """Test that similar texts score higher than different texts."""
        results_similar = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )
        results_different = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            rouge_type=RougeType.ROUGE_1,
        )

        result_similar = self._extract_and_print_result(results_similar, "Similar Text")
        result_different = self._extract_and_print_result(results_different, "Different Text")

        assert result_similar["f1_score"] > result_different["f1_score"]
