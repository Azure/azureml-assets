# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for GLEU Score Evaluator."""

import pytest
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.gleu_score.evaluator._gleu import GleuScoreEvaluator


@pytest.mark.unittest
class TestGleuScoreEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for GLEU Score Evaluator.

    Tests different input formats, thresholds, and edge cases.
    GLEU (Google-BLEU) evaluates n-gram overlap considering both precision and recall.
    It is designed for sentence-level assessment of translation quality.
    Scores range from 0 to 1, with 1 being perfect overlap.
    """

    evaluator_type = GleuScoreEvaluator
    result_key = "gleu_score"

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

    # Case sensitivity
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

    # N-gram specific tests
    BIGRAM_TEST_REF = "the cat sat"
    BIGRAM_TEST_SIMILAR = "the cat sat down"
    BIGRAM_TEST_DIFFERENT = "sat cat the"  # Same words but different n-grams

    # Word order matters for GLEU (n-gram based)
    ORDERED_TEXT = "apple banana cherry date"
    REVERSED_TEXT = "date cherry banana apple"
    # endregion

    # ==================== PERFECT MATCH TESTS ====================

    def test_identical_text(self):
        """Test with identical response and ground truth (should have high GLEU score)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Text")
        self.assert_pass(result_data)
        # Identical text should have perfect or near-perfect GLEU score
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
        # Single identical word should have high score
        assert result_data["score"] >= 0.5

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
        # Similar texts should have moderate to high GLEU score
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
        # Different texts should have low GLEU score
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

    # ==================== N-GRAM ORDER TESTS ====================

    def test_word_order_matters(self):
        """Test that GLEU considers word order (n-gram based)."""
        # Same words but reversed order should have lower score
        results_ordered = self._run_evaluation(
            response=self.ORDERED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            threshold=0.5,
        )
        results_reversed = self._run_evaluation(
            response=self.REVERSED_TEXT,
            ground_truth=self.ORDERED_TEXT,
            threshold=0.5,
        )
        result_ordered = self._extract_and_print_result(results_ordered, "Ordered Text")
        result_reversed = self._extract_and_print_result(results_reversed, "Reversed Text")

        # Ordered should have perfect score
        assert result_ordered["score"] >= 0.9
        # Reversed should have lower score due to n-gram mismatch
        assert result_reversed["score"] < result_ordered["score"]

    def test_bigram_overlap(self):
        """Test bigram overlap scenarios."""
        # Same words, different bigrams
        results = self._run_evaluation(
            response=self.BIGRAM_TEST_DIFFERENT,
            ground_truth=self.BIGRAM_TEST_REF,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Bigram Overlap")
        self.assert_score_in_range(result_data)
        # Score should be lower than identical due to bigram mismatch

    def test_bigram_similar(self):
        """Test similar texts with shared bigrams."""
        results = self._run_evaluation(
            response=self.BIGRAM_TEST_SIMILAR,
            ground_truth=self.BIGRAM_TEST_REF,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Bigram Similar")
        self.assert_score_in_range(result_data)
        # Should have good score since bigrams like "the cat", "cat sat" are preserved

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
        evaluator = GleuScoreEvaluator()
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
        # Empty response should have zero GLEU score
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
        # Empty ground truth should result in zero GLEU score
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
        # Should have moderate to good score since reference n-grams are present

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
        assert "gleu_score" in results
        assert "gleu_result" in results
        assert "gleu_threshold" in results

    def test_output_score_type(self):
        """Test that gleu_score is a float."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert isinstance(results["gleu_score"], float)

    def test_output_result_values(self):
        """Test that gleu_result is either 'pass' or 'fail'."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert results["gleu_result"] in ["pass", "fail"]

    def test_output_threshold_matches_input(self):
        """Test that output threshold matches input threshold."""
        threshold = 0.75
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=threshold,
        )
        assert results["gleu_threshold"] == threshold

    # ==================== GLEU SCORE RANGE TESTS ====================

    def test_gleu_score_range_various_inputs(self):
        """Test that GLEU score is always in [0, 1] range."""
        test_cases = [
            (self.IDENTICAL_TEXT, self.IDENTICAL_TEXT),
            (self.DIFFERENT_RESPONSE, self.REFERENCE_TEXT),
            (self.EMPTY_STRING, self.REFERENCE_TEXT),
            (self.LONG_RESPONSE, self.SINGLE_WORD),
            (self.PARTIAL_MATCH_RESPONSE, self.PARTIAL_MATCH_REFERENCE),
        ]
        for response, ground_truth in test_cases:
            results = self._run_evaluation(
                response=response,
                ground_truth=ground_truth,
                threshold=0.5,
            )
            if "gleu_score" in results:
                assert 0.0 <= results["gleu_score"] <= 1.0

    # ==================== COMPARISON WITH EXPECTED BEHAVIOR ====================

    def test_gleu_vs_bleu_behavior(self):
        """Test that GLEU behaves as expected (balances precision and recall)."""
        # GLEU should give credit for both precision and recall
        # Unlike BLEU which focuses mainly on precision

        # Short response with perfect precision but low recall
        short_response = "cat"
        long_ground_truth = "the cat sat on the mat in the house"

        results = self._run_evaluation(
            response=short_response,
            ground_truth=long_ground_truth,
            threshold=0.1,
        )
        result_data = self._extract_and_print_result(results, "GLEU Short Response")
        self.assert_score_in_range(result_data)

    def test_symmetric_behavior(self):
        """Test GLEU behavior when swapping response and ground truth."""
        results1 = self._run_evaluation(
            response="cat sat mat",
            ground_truth="the cat sat on the mat",
            threshold=0.3,
        )
        results2 = self._run_evaluation(
            response="the cat sat on the mat",
            ground_truth="cat sat mat",
            threshold=0.3,
        )
        result1 = self._extract_and_print_result(results1, "GLEU Order 1")
        result2 = self._extract_and_print_result(results2, "GLEU Order 2")

        # Both should have valid scores (may not be identical due to GLEU's formula)
        self.assert_score_in_range(result1)
        self.assert_score_in_range(result2)

    # ==================== TOKENIZATION TESTS ====================

    def test_tokenization_handles_contractions(self):
        """Test that tokenization handles contractions."""
        results = self._run_evaluation(
            response="I'm going to the store",
            ground_truth="I'm going to the store",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Contractions")
        self.assert_pass(result_data)

    def test_tokenization_handles_hyphenated_words(self):
        """Test that tokenization handles hyphenated words."""
        results = self._run_evaluation(
            response="well-known fact",
            ground_truth="well-known fact",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Hyphenated Words")
        self.assert_pass(result_data)

    def test_tokenization_handles_numbers_with_text(self):
        """Test tokenization with mixed numbers and text."""
        results = self._run_evaluation(
            response="I have 3 apples and 2 oranges",
            ground_truth="I have 3 apples and 2 oranges",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numbers with Text")
        self.assert_pass(result_data)
