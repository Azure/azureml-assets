# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Similarity Evaluator."""

import pytest
from typing import List
from ...builtin.similarity.evaluator._similarity import SimilarityEvaluator
from ..common import BasePromptyEvaluatorRunner


@pytest.mark.unittest
class TestSimilarityEvaluatorBehavior(BasePromptyEvaluatorRunner):
    """
    Behavioral tests for Similarity Evaluator.

    Tests different input formats, thresholds, and scenarios.
    The Similarity evaluator measures semantic similarity between response and ground_truth
    given a query. It uses an LLM to produce scores from 1 to 5:
    - 1: Not at all similar
    - 2: Mostly not similar
    - 3: Somewhat similar
    - 4: Mostly similar
    - 5: Completely similar

    Default threshold is 3.
    """

    evaluator_type = SimilarityEvaluator
    use_mocking = True

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for prompty evaluators."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_threshold"
        ]

    constructor_arg_names = ["threshold"]

    # region Test Data
    # Query examples
    VALID_QUERY = "What is the role of ribosomes?"
    WEATHER_QUERY = "What is the weather like in Seattle?"
    MATH_QUERY = "What is 2 + 2?"

    # Perfect match scenario
    PERFECT_GROUND_TRUTH = "Ribosomes are cellular structures responsible for protein synthesis."
    PERFECT_RESPONSE = "Ribosomes are cellular structures responsible for protein synthesis."

    # High similarity scenario (score 5)
    HIGH_SIM_GROUND_TRUTH = "Regular exercise can help maintain a healthy weight, increase muscle strength."
    HIGH_SIM_RESPONSE = (
        "Routine physical activity can contribute to maintaining ideal body weight, enhancing muscle strength."
    )

    # Moderate similarity scenario (score 3-4)
    MODERATE_SIM_GROUND_TRUTH = "The Earth orbits the Sun, causing seasons due to axial tilt."
    MODERATE_SIM_RESPONSE = "Seasons occur because of the Earth's rotation and orbital path around the Sun."

    # Low similarity scenario (score 1-2)
    LOW_SIM_GROUND_TRUTH = "Ribosomes are responsible for protein synthesis."
    LOW_SIM_RESPONSE = "Ribosomes participate in carbohydrate breakdown."

    # No similarity scenario (score 1)
    NO_SIM_GROUND_TRUTH = "The capital of France is Paris."
    NO_SIM_RESPONSE = "Elephants are the largest land mammals."

    # Edge cases
    EMPTY_STRING = ""
    SINGLE_WORD = "Hello"
    LONG_TEXT = "This is a very long text that contains many words and sentences. " * 20
    UNICODE_TEXT = "こんにちは世界"
    SPECIAL_CHARS = "Hello! How are you? I'm fine, thanks."

    # Multi-sentence scenarios
    MULTI_SENTENCE_GROUND_TRUTH = "The Titanic sank in 1912. It hit an iceberg. Many passengers died."
    MULTI_SENTENCE_RESPONSE = "The Titanic struck an iceberg in 1912 and sank, resulting in many deaths."

    # Technical/domain-specific
    TECHNICAL_GROUND_TRUTH = "Photosynthesis converts light energy into chemical energy in chloroplasts."
    TECHNICAL_RESPONSE = "Light energy is transformed into chemical energy through photosynthesis in chloroplasts."

    # Paraphrase scenarios
    PARAPHRASE_GROUND_TRUTH = "The quick brown fox jumps over the lazy dog."
    PARAPHRASE_RESPONSE = "A fast brown fox leaps over a sleepy dog."
    # endregion

    # ==================== ALL VALID INPUTS TEST ====================

    def test_all_valid_inputs(self):
        """All inputs valid and in correct format - should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "All Valid")
        self.assert_pass(result_data)

    # ==================== PERFECT SIMILARITY TESTS ====================

    def test_identical_response_and_ground_truth(self):
        """Test with identical response and ground truth (should score 5)."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Identical Response and Ground Truth")
        self.assert_pass(result_data)

    def test_high_similarity_paraphrase(self):
        """Test with highly similar paraphrased response (should score 5)."""
        results = self._run_evaluation(
            query="What are the health benefits of exercise?",
            response=self.HIGH_SIM_RESPONSE,
            ground_truth=self.HIGH_SIM_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "High Similarity Paraphrase")
        self.assert_pass(result_data)

    # ==================== MODERATE SIMILARITY TESTS ====================

    def test_moderate_similarity(self):
        """Test with moderately similar response - should pass with mocked score."""
        results = self._run_evaluation(
            query="What causes seasons on Earth?",
            response=self.MODERATE_SIM_RESPONSE,
            ground_truth=self.MODERATE_SIM_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Moderate Similarity")
        self.assert_pass(result_data)

    def test_moderate_similarity_score_4(self):
        """Test with moderately similar technical response - should pass with mocked score."""
        results = self._run_evaluation(
            query="How does photosynthesis work?",
            response=self.TECHNICAL_RESPONSE,
            ground_truth=self.TECHNICAL_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Moderate Similarity Score 4")
        self.assert_pass(result_data)

    # ==================== MULTI-SENTENCE TESTS ====================

    def test_multi_sentence_response(self):
        """Test with multi-sentence response and ground truth - valid input should pass."""
        results = self._run_evaluation(
            query="What happened to the Titanic?",
            response=self.MULTI_SENTENCE_RESPONSE,
            ground_truth=self.MULTI_SENTENCE_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Multi-Sentence")
        self.assert_pass(result_data)

    # ==================== PARAPHRASE TESTS ====================

    def test_paraphrase_detection(self):
        """Test that paraphrases are recognized as similar - valid input should pass."""
        results = self._run_evaluation(
            query="Describe the animal action.",
            response=self.PARAPHRASE_RESPONSE,
            ground_truth=self.PARAPHRASE_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Paraphrase Detection")
        self.assert_pass(result_data)

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string - evaluator accepts it, mock returns score 5."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.EMPTY_STRING,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Empty Response")
        self.assert_pass(result_data)

    def test_empty_ground_truth(self):
        """Test with empty ground truth string - evaluator accepts it, mock returns score 5."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.EMPTY_STRING,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        self.assert_pass(result_data)

    def test_empty_query(self):
        """Test with empty query string - evaluator accepts it, mock returns score 5."""
        results = self._run_evaluation(
            query=self.EMPTY_STRING,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Empty Query")
        self.assert_pass(result_data)

    def test_single_word_inputs(self):
        """Test with single word inputs - evaluator accepts them."""
        results = self._run_evaluation(
            query="Word?",
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
        )
        result_data = self._extract_and_print_result(results, "Single Word")
        self.assert_pass(result_data)

    def test_long_text_inputs(self):
        """Test with very long text inputs - evaluator accepts them."""
        results = self._run_evaluation(
            query="Describe something in detail.",
            response=self.LONG_TEXT,
            ground_truth=self.LONG_TEXT,
        )
        result_data = self._extract_and_print_result(results, "Long Text")
        self.assert_pass(result_data)

    def test_unicode_text(self):
        """Test with Unicode characters - evaluator accepts them."""
        results = self._run_evaluation(
            query="Translate to Japanese.",
            response=self.UNICODE_TEXT,
            ground_truth=self.UNICODE_TEXT,
        )
        result_data = self._extract_and_print_result(results, "Unicode Text")
        self.assert_pass(result_data)

    def test_special_characters(self):
        """Test with special characters and punctuation - evaluator accepts them."""
        results = self._run_evaluation(
            query="How are you?",
            response=self.SPECIAL_CHARS,
            ground_truth=self.SPECIAL_CHARS,
        )
        result_data = self._extract_and_print_result(results, "Special Characters")
        self.assert_pass(result_data)

    # ==================== ERROR HANDLING TESTS ====================

    def test_none_query(self):
        """Test with None query."""
        results = self._run_evaluation(
            query=None,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "None Query")
        self.assert_missing_field_error(result_data)

    def test_none_response(self):
        """Test with None response."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=None,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "None Response")
        self.assert_missing_field_error(result_data)

    def test_none_ground_truth(self):
        """Test with None ground truth."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=None,
        )
        result_data = self._extract_and_print_result(results, "None Ground Truth")
        self.assert_missing_field_error(result_data)

    def test_numeric_query_type(self):
        """Test with numeric query type (int) - evaluator converts to string, should pass."""
        results = self._run_evaluation(
            query=12345,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Numeric Query Type")
        self.assert_pass(result_data)

    def test_list_response_type(self):
        """Test with list response type - evaluator converts to string, should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=["hello", "world"],
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "List Response Type")
        self.assert_pass(result_data)

    def test_dict_ground_truth_type(self):
        """Test with dict ground truth type - evaluator converts to string, should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth={"text": "hello"},
        )
        result_data = self._extract_and_print_result(results, "Dict Ground Truth Type")
        self.assert_pass(result_data)

    # ==================== OUTPUT STRUCTURE TESTS ====================

    def test_output_contains_required_keys(self):
        """Test that output contains all required keys."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        assert "similarity" in results
        assert "similarity_result" in results
        assert "similarity_threshold" in results

    def test_output_score_type(self):
        """Test that similarity score is numeric."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        assert isinstance(results["similarity"], (int, float))

    def test_output_result_values(self):
        """Test that similarity_result is either 'pass' or 'fail'."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        assert results["similarity_result"] in ["pass", "fail"]

    # ==================== SEMANTIC EQUIVALENCE TESTS ====================

    def test_semantic_equivalence_same_meaning(self):
        """Test that semantically equivalent responses are accepted as valid input."""
        results = self._run_evaluation(
            query="What is H2O?",
            response="H2O is the chemical formula for water.",
            ground_truth="Water has the chemical formula H2O.",
        )
        result_data = self._extract_and_print_result(results, "Semantic Equivalence")
        self.assert_pass(result_data)

    # ==================== DOMAIN-SPECIFIC TESTS ====================

    def test_technical_domain_similarity(self):
        """Test with technical/scientific content - valid input should pass."""
        results = self._run_evaluation(
            query="Explain DNA replication.",
            response="DNA replication is the process of copying genetic material.",
            ground_truth="DNA replication copies the genetic information stored in DNA.",
        )
        result_data = self._extract_and_print_result(results, "Technical Domain")
        self.assert_pass(result_data)

    def test_code_like_content(self):
        """Test with code-like content - valid input should pass."""
        results = self._run_evaluation(
            query="Write a hello world function.",
            response="def hello(): print('Hello')",
            ground_truth="def hello_world(): print('Hello, World!')",
        )
        result_data = self._extract_and_print_result(results, "Code Content")
        self.assert_pass(result_data)

    # ==================== THRESHOLD TESTS ====================

    def test_custom_threshold_1(self):
        """Test with minimum threshold of 1 - any score should pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.LOW_SIM_RESPONSE,
            ground_truth=self.LOW_SIM_GROUND_TRUTH,
            threshold=1,
        )
        result_data = self._extract_and_print_result(results, "Threshold 1")
        # With threshold=1, even low similarity should pass
        self.assert_pass(result_data)

    def test_custom_threshold_5(self):
        """Test with maximum threshold of 5 - only perfect scores pass."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=5,
        )
        result_data = self._extract_and_print_result(results, "Threshold 5")
        # Perfect match should still pass with threshold=5
        self.assert_pass(result_data)

    def test_threshold_boundary_at_3(self):
        """Test default threshold boundary - score 5 from mock >= threshold 3 should pass."""
        results = self._run_evaluation(
            query="What causes seasons on Earth?",
            response=self.MODERATE_SIM_RESPONSE,
            ground_truth=self.MODERATE_SIM_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Threshold Boundary")
        self.assert_pass(result_data)

    def test_threshold_in_output(self):
        """Test that custom threshold is reflected in output."""
        custom_threshold = 4
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=custom_threshold,
        )
        assert results["similarity_threshold"] == custom_threshold

    def test_default_threshold_is_3(self):
        """Test that default threshold is 3."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        assert results["similarity_threshold"] == 3

    # ==================== SCORE RANGE VALIDATION TESTS ====================

    def test_score_is_in_valid_range(self):
        """Test that similarity score is between 1 and 5."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        score = results["similarity"]
        assert 1 <= score <= 5, f"Score {score} is outside valid range [1, 5]"

    def test_score_is_number(self):
        """Test that similarity score is a number (1-5 scale)."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        score = results["similarity"]
        assert isinstance(score, (int, float)), f"Score {score} should be a number"

    # ==================== INVALID THRESHOLD TESTS ====================

    # TODO: Determine whether we should add validation for threshold range (e.g., 1-5).
    # Currently, thresholds outside the score range are accepted without error.

    def test_threshold_below_minimum(self):
        """Test with threshold below valid range (0) - currently accepted without validation."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=0,
        )
        result_data = self._extract_and_print_result(results, "Threshold Below Min")
        self.assert_pass(result_data)

    def test_threshold_above_maximum(self):
        """Test with threshold above valid range (6) - currently accepted without validation."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=6,
        )
        result_data = self._extract_and_print_result(results, "Threshold Above Max")
        self.assert_fail(result_data)

    def test_threshold_negative(self):
        """Test with negative threshold - currently accepted without validation."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=-1,
        )
        result_data = self._extract_and_print_result(results, "Threshold Negative")
        self.assert_pass(result_data)

    def test_threshold_float(self):
        """Test with float threshold - floats are valid threshold values."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=2.5,
        )
        result_data = self._extract_and_print_result(results, "Threshold Float")
        self.assert_pass(result_data)

    def test_threshold_string_type(self):
        """Test with string threshold - should raise error."""
        # TODO: Should we add type validation for threshold to ensure it's a number?
        # Currently, it throws code error due to unsupported comparison between str and int when determining pass/fail
        #  but we may want to catch this earlier with a clearer error message.
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold="3",
        )
        result_data = self._extract_and_print_result(results, "Threshold String Type")
        self.assert_error(result_data, error_code=None)  # Currently no validation, so error_code is None

    # ==================== COMBINED EDGE CASE TESTS ====================

    def test_all_none_inputs(self):
        """Test with all inputs as None - should raise missing field error for query first."""
        results = self._run_evaluation(
            query=None,
            response=None,
            ground_truth=None,
        )
        result_data = self._extract_and_print_result(results, "All None Inputs")
        self.assert_missing_field_error(result_data)

    def test_all_empty_inputs(self):
        """Test with all inputs as empty strings - evaluator accepts them."""
        results = self._run_evaluation(
            query=self.EMPTY_STRING,
            response=self.EMPTY_STRING,
            ground_truth=self.EMPTY_STRING,
        )
        result_data = self._extract_and_print_result(results, "All Empty Inputs")
        self.assert_pass(result_data)

    def test_whitespace_only_inputs(self):
        """Test with whitespace-only strings - evaluator accepts them."""
        results = self._run_evaluation(
            query="   ",
            response="   \t\n  ",
            ground_truth="   ",
        )
        result_data = self._extract_and_print_result(results, "Whitespace Only")
        self.assert_pass(result_data)

    # ==================== ADDITIONAL OUTPUT STRUCTURE TESTS ====================

    def test_output_pass_label_when_score_equals_threshold(self):
        """Test that result is 'pass' when threshold equals max valid score."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
            threshold=5,
        )
        assert results["similarity_result"] == "pass"
        assert results["similarity"] >= results["similarity_threshold"]
