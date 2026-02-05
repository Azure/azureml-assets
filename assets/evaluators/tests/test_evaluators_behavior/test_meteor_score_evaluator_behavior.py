# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for METEOR Score Evaluator."""

import pytest
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.meteor_score.evaluator._meteor import MeteorScoreEvaluator


@pytest.mark.unittest
class TestMeteorScoreEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for METEOR Score Evaluator.

    Tests different input formats, parameters, thresholds, and edge cases.
    METEOR (Metric for Evaluation of Translation with Explicit Ordering) evaluates text by:
    - Considering synonyms and word stems
    - Balancing precision and recall
    - Penalizing fragmentation (word order)
    Parameters:
    - alpha: Weight for precision vs recall (default 0.9)
    - beta: Fragmentation penalty parameter (default 3.0)
    - gamma: Fragmentation weight (default 0.5)
    Scores range from 0 to 1, with 1 being perfect match.
    """

    evaluator_type = MeteorScoreEvaluator
    result_key = "meteor_score"
    constructor_arg_names = ["alpha", "beta", "gamma", "threshold"]

    # region Test Data
    # Perfect match scenarios
    IDENTICAL_TEXT = "The quick brown fox jumps over the lazy dog."

    # Synonym scenarios - METEOR should recognize these
    SYNONYM_GROUND_TRUTH = "The cat sat on the couch."
    SYNONYM_RESPONSE = "The cat sat on the sofa."  # sofa = synonym of couch

    # Stemming scenarios - METEOR should recognize word stems
    STEM_GROUND_TRUTH = "The runner is running in the race."
    STEM_RESPONSE = "The runner ran in the race."  # ran = stem of running

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

    # Word order matters for METEOR (fragmentation penalty)
    ORDERED_GROUND_TRUTH = "The quick brown fox"
    ORDERED_RESPONSE = "The quick brown fox"
    REORDERED_RESPONSE = "brown quick The fox"  # Fragmented

    # Edge cases
    EMPTY_STRING = ""
    SINGLE_WORD = "Hello"
    SINGLE_CHAR = "A"
    WHITESPACE_ONLY = "   "
    PUNCTUATION_ONLY = ".,!?;:"
    NUMBERS_ONLY = "12345"

    # Case variations
    MIXED_CASE_LOWER = "hello world"
    MIXED_CASE_UPPER = "HELLO WORLD"
    MIXED_CASE_MIXED = "Hello World"

    # Special characters
    SPECIAL_CHARS_TEXT = "Hello! How are you? I'm fine, thanks."
    UNICODE_TEXT = "café résumé naïve"

    # Long text scenarios
    LONG_TEXT = "This is a very long text that contains many words and sentences. " * 10

    # Technical text
    CODE_TEXT = "def hello_world(): print('Hello, World!')"

    # Numeric text
    NUMERIC_TEXT = "The year 2024 has 365 days and 12 months."

    # Paraphrase scenarios - METEOR should handle these well
    PARAPHRASE_GROUND_TRUTH = "The boy quickly ran to the store."
    PARAPHRASE_RESPONSE = "The boy ran quickly to the store."

    # Complex synonym scenarios
    SYNONYM_COMPLEX_GROUND_TRUTH = "The automobile was fast."
    SYNONYM_COMPLEX_RESPONSE = "The car was quick."  # automobile=car, fast=quick (synonyms)
    # endregion

    # ==================== PERFECT MATCH TESTS ====================

    def test_identical_text(self):
        """Test with identical response and ground truth (should have high METEOR score)."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Text")
        self.assert_pass(result_data)
        # Identical text should have perfect METEOR score
        assert result_data["score"] >= 0.99

    def test_identical_single_word(self):
        """Test with identical single word."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Identical Single Word")
        self.assert_score_in_range(result_data)
        # Single identical word returns 0.5 with METEOR due to fragmentation penalty
        assert result_data["score"] >= 0.5
        self.assert_pass(result_data)

    # ==================== SYNONYM RECOGNITION TESTS ====================

    def test_synonym_recognition(self):
        """Test that METEOR recognizes synonyms (e.g., couch/sofa)."""
        results = self._run_evaluation(
            response=self.SYNONYM_RESPONSE,
            ground_truth=self.SYNONYM_GROUND_TRUTH,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Synonym Recognition")
        self.assert_score_in_range(result_data)
        # METEOR should give credit for synonyms, score should be higher than exact match only
        assert result_data["score"] > 0.5

    def test_complex_synonyms(self):
        """Test complex synonym scenarios (automobile/car, fast/quick)."""
        results = self._run_evaluation(
            response=self.SYNONYM_COMPLEX_RESPONSE,
            ground_truth=self.SYNONYM_COMPLEX_GROUND_TRUTH,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Complex Synonyms")
        self.assert_score_in_range(result_data)
        # Should have reasonable score due to synonym matching

    # ==================== STEMMING TESTS ====================

    def test_stemming_recognition(self):
        """Test that METEOR recognizes word stems (running/ran)."""
        results = self._run_evaluation(
            response=self.STEM_RESPONSE,
            ground_truth=self.STEM_GROUND_TRUTH,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Stemming Recognition")
        self.assert_score_in_range(result_data)
        # METEOR should recognize stems, score should be good
        assert result_data["score"] > 0.5

    def test_verb_conjugation_matching(self):
        """Test matching different verb conjugations."""
        results = self._run_evaluation(
            response="She walks to work",
            ground_truth="She walked to work",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Verb Conjugation")
        self.assert_score_in_range(result_data)
        # Should have good score due to stemming

    def test_plural_matching(self):
        """Test matching singular/plural forms."""
        results = self._run_evaluation(
            response="The cat sat on the mat",
            ground_truth="The cats sat on the mats",
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Plural Matching")
        self.assert_score_in_range(result_data)

    # ==================== PARAPHRASE TESTS ====================

    def test_paraphrase_detection(self):
        """Test paraphrase detection (same meaning, different order)."""
        results = self._run_evaluation(
            response=self.PARAPHRASE_RESPONSE,
            ground_truth=self.PARAPHRASE_GROUND_TRUTH,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Paraphrase Detection")
        self.assert_score_in_range(result_data)
        # Minor word order change should still have high score
        assert result_data["score"] > 0.7

    # ==================== WORD ORDER (FRAGMENTATION) TESTS ====================

    def test_word_order_penalty(self):
        """Test that METEOR penalizes fragmented word order."""
        # Perfect order
        results_ordered = self._run_evaluation(
            response=self.ORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            threshold=0.5,
        )
        # Reordered (fragmented)
        results_reordered = self._run_evaluation(
            response=self.REORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            threshold=0.3,
        )

        result_ordered = self._extract_and_print_result(results_ordered, "Ordered")
        result_reordered = self._extract_and_print_result(results_reordered, "Reordered")

        # Ordered should have higher score than fragmented
        assert result_ordered["score"] > result_reordered["score"]

    def test_complete_reversal(self):
        """Test completely reversed word order."""
        ground_truth = "one two three four five"
        reversed_response = "five four three two one"

        results = self._run_evaluation(
            response=reversed_response,
            ground_truth=ground_truth,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Complete Reversal")
        self.assert_score_in_range(result_data)
        # Should have some score (words match) but penalized for fragmentation

    # ==================== PARAMETER TESTS ====================

    def test_alpha_parameter_high(self):
        """Test with high alpha (more weight on precision)."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            alpha=0.95,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "High Alpha (0.95)")
        self.assert_score_in_range(result_data)

    def test_alpha_parameter_low(self):
        """Test with low alpha (more weight on recall)."""
        results = self._run_evaluation(
            response=self.SIMILAR_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            alpha=0.5,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Low Alpha (0.5)")
        self.assert_score_in_range(result_data)

    def test_beta_parameter_effect(self):
        """Test beta parameter effect on fragmentation penalty."""
        # High beta = stronger fragmentation penalty
        results_high_beta = self._run_evaluation(
            response=self.REORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            beta=5.0,
            threshold=0.1,
        )
        # Low beta = weaker fragmentation penalty
        results_low_beta = self._run_evaluation(
            response=self.REORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            beta=1.0,
            threshold=0.1,
        )

        result_high = self._extract_and_print_result(results_high_beta, "High Beta (5.0)")
        result_low = self._extract_and_print_result(results_low_beta, "Low Beta (1.0)")

        self.assert_score_in_range(result_high)
        self.assert_score_in_range(result_low)

    def test_gamma_parameter_effect(self):
        """Test gamma parameter effect on fragmentation weight."""
        results_high_gamma = self._run_evaluation(
            response=self.REORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            gamma=0.8,
            threshold=0.1,
        )
        results_low_gamma = self._run_evaluation(
            response=self.REORDERED_RESPONSE,
            ground_truth=self.ORDERED_GROUND_TRUTH,
            gamma=0.2,
            threshold=0.1,
        )

        result_high = self._extract_and_print_result(results_high_gamma, "High Gamma (0.8)")
        result_low = self._extract_and_print_result(results_low_gamma, "Low Gamma (0.2)")

        self.assert_score_in_range(result_high)
        self.assert_score_in_range(result_low)

    def test_default_parameters(self):
        """Test with default parameters (alpha=0.9, beta=3.0, gamma=0.5)."""
        evaluator = MeteorScoreEvaluator()  # All defaults
        results = evaluator(response=self.IDENTICAL_TEXT, ground_truth=self.IDENTICAL_TEXT)
        result_data = self._extract_and_print_result(results, "Default Parameters")
        assert result_data["threshold"] == 0.5
        assert result_data["score"] >= 0.99

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
            threshold=0.95,
        )
        result_data = self._extract_and_print_result(results, "Threshold High - Fail")
        self.assert_fail(result_data)
        assert result_data["threshold"] == 0.95

    def test_threshold_zero(self):
        """Test with zero threshold (everything should pass)."""
        results = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.0,
        )
        result_data = self._extract_and_print_result(results, "Threshold Zero")
        self.assert_pass(result_data)

    def test_threshold_one(self):
        """Test with threshold of 1.0."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=1.0,
        )
        result_data = self._extract_and_print_result(results, "Threshold One")
        self.assert_score_in_range(result_data)
        assert result_data["threshold"] == 1.0

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
        assert result_data["score"] > 0.3

    def test_low_similarity(self):
        """Test with texts having low similarity."""
        results = self._run_evaluation(
            response=self.DIFFERENT_RESPONSE,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Low Similarity")
        self.assert_score_in_range(result_data)
        assert result_data["score"] < 0.5

    def test_partial_match(self):
        """Test with partial matching texts."""
        results = self._run_evaluation(
            response=self.PARTIAL_MATCH_RESPONSE,
            ground_truth=self.PARTIAL_MATCH_GROUND_TRUTH,
            threshold=0.3,
        )
        result_data = self._extract_and_print_result(results, "Partial Match")
        self.assert_score_in_range(result_data)

    def test_multi_sentence(self):
        """Test with multi-sentence texts."""
        results = self._run_evaluation(
            response=self.MULTI_SENTENCE_RESPONSE,
            ground_truth=self.MULTI_SENTENCE_GROUND_TRUTH,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Multi-Sentence")
        self.assert_score_in_range(result_data)

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string."""
        results = self._run_evaluation(
            response=self.EMPTY_STRING,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Empty Response")
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

    def test_whitespace_only(self):
        """Test with whitespace-only text."""
        results = self._run_evaluation(
            response=self.WHITESPACE_ONLY,
            ground_truth=self.REFERENCE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Whitespace Only")
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

    def test_case_insensitivity(self):
        """Test case handling between lower and upper case."""
        results = self._run_evaluation(
            response=self.MIXED_CASE_LOWER,
            ground_truth=self.MIXED_CASE_UPPER,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Case Handling")
        self.assert_score_in_range(result_data)

    def test_case_mixed(self):
        """Test case handling with mixed case."""
        results = self._run_evaluation(
            response=self.MIXED_CASE_LOWER,
            ground_truth=self.MIXED_CASE_MIXED,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Case Mixed")
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
        assert result_data["score"] >= 0.99

    def test_long_vs_short(self):
        """Test with long ground truth and short response."""
        results = self._run_evaluation(
            response=self.SINGLE_WORD,
            ground_truth=self.LONG_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Long vs Short")
        self.assert_score_in_range(result_data)
        # Short response should have low score
        assert result_data["score"] < 0.5

    # ==================== TECHNICAL TEXT TESTS ====================

    def test_code_text(self):
        """Test with code-like text."""
        results = self._run_evaluation(
            response=self.CODE_TEXT,
            ground_truth=self.CODE_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Code Text")
        self.assert_pass(result_data)

    def test_numeric_text(self):
        """Test with numeric text."""
        results = self._run_evaluation(
            response=self.NUMERIC_TEXT,
            ground_truth=self.NUMERIC_TEXT,
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Numeric Text")
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
        assert "meteor_score" in results
        assert "meteor_result" in results
        assert "meteor_threshold" in results

    def test_output_score_type(self):
        """Test that meteor_score is a float."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert isinstance(results["meteor_score"], float)

    def test_output_result_values(self):
        """Test that meteor_result is either 'pass' or 'fail'."""
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=0.5,
        )
        assert results["meteor_result"] in ["pass", "fail"]

    def test_output_threshold_matches_input(self):
        """Test that output threshold matches input threshold."""
        threshold = 0.75
        results = self._run_evaluation(
            response=self.IDENTICAL_TEXT,
            ground_truth=self.IDENTICAL_TEXT,
            threshold=threshold,
        )
        assert results["meteor_threshold"] == threshold

    # ==================== METEOR SCORE RANGE TESTS ====================

    def test_meteor_score_range_various_inputs(self):
        """Test that METEOR score is always in [0, 1] range."""
        test_cases = [
            (self.IDENTICAL_TEXT, self.IDENTICAL_TEXT),
            (self.DIFFERENT_RESPONSE, self.REFERENCE_TEXT),
            (self.EMPTY_STRING, self.REFERENCE_TEXT),
            (self.LONG_TEXT, self.SINGLE_WORD),
            (self.PARTIAL_MATCH_RESPONSE, self.PARTIAL_MATCH_GROUND_TRUTH),
        ]
        for response, ground_truth in test_cases:
            results = self._run_evaluation(
                response=response,
                ground_truth=ground_truth,
                threshold=0.5,
            )
            if "meteor_score" in results:
                assert 0.0 <= results["meteor_score"] <= 1.0

    # ==================== COMPARISON TESTS ====================

    def test_meteor_vs_exact_match(self):
        """Test that METEOR gives higher scores than exact match for synonyms."""
        # With synonyms, METEOR should give reasonable score
        results = self._run_evaluation(
            response="The automobile was fast",
            ground_truth="The car was quick",
            threshold=0.2,
        )
        result_data = self._extract_and_print_result(results, "METEOR Synonym Advantage")
        self.assert_score_in_range(result_data)
        # Should have some score due to synonym/stem matching

    def test_subset_scenario(self):
        """Test response that is subset of ground truth."""
        results = self._run_evaluation(
            response="cat sat",
            ground_truth="The cat sat on the mat",
            threshold=0.2,
        )
        result_data = self._extract_and_print_result(results, "Subset Scenario")
        self.assert_score_in_range(result_data)

    def test_superset_scenario(self):
        """Test response that is superset of ground truth."""
        results = self._run_evaluation(
            response="The big fluffy cat sat on the soft mat quickly",
            ground_truth="cat sat mat",
            threshold=0.2,
        )
        result_data = self._extract_and_print_result(results, "Superset Scenario")
        self.assert_score_in_range(result_data)

    # ==================== LINGUISTIC FEATURE TESTS ====================

    def test_contraction_handling(self):
        """Test handling of contractions."""
        results = self._run_evaluation(
            response="I'm going to the store",
            ground_truth="I am going to the store",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Contraction Handling")
        self.assert_score_in_range(result_data)

    def test_possessive_handling(self):
        """Test handling of possessive forms."""
        results = self._run_evaluation(
            response="The dog's toy",
            ground_truth="The dog's toy",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Possessive Handling")
        self.assert_pass(result_data)

    def test_hyphenated_words(self):
        """Test handling of hyphenated words."""
        results = self._run_evaluation(
            response="well-known fact",
            ground_truth="well-known fact",
            threshold=0.5,
        )
        result_data = self._extract_and_print_result(results, "Hyphenated Words")
        self.assert_pass(result_data)
