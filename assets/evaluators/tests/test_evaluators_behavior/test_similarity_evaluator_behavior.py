# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Similarity Evaluator."""

import pytest

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
        """Test with moderately similar response (should score 3-4)."""
        results = self._run_evaluation(
            query="What causes seasons on Earth?",
            response=self.MODERATE_SIM_RESPONSE,
            ground_truth=self.MODERATE_SIM_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Moderate Similarity")
        self.assert_pass_or_fail(result_data)

    def test_moderate_similarity_score_4(self):
        """Test with moderately similar response scoring 4."""
        results = self._run_evaluation(
            query="How does photosynthesis work?",
            response=self.TECHNICAL_RESPONSE,
            ground_truth=self.TECHNICAL_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Moderate Similarity Score 4")
        self.assert_pass_or_fail(result_data)

    # ==================== MULTI-SENTENCE TESTS ====================

    def test_multi_sentence_response(self):
        """Test with multi-sentence response and ground truth."""
        results = self._run_evaluation(
            query="What happened to the Titanic?",
            response=self.MULTI_SENTENCE_RESPONSE,
            ground_truth=self.MULTI_SENTENCE_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Multi-Sentence")
        self.assert_pass_or_fail(result_data)

    # ==================== PARAPHRASE TESTS ====================

    def test_paraphrase_detection(self):
        """Test that paraphrases are recognized as similar."""
        results = self._run_evaluation(
            query="Describe the animal action.",
            response=self.PARAPHRASE_RESPONSE,
            ground_truth=self.PARAPHRASE_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Paraphrase Detection")
        self.assert_pass_or_fail(result_data)

    # ==================== EDGE CASE TESTS ====================

    def test_empty_response(self):
        """Test with empty response string."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.EMPTY_STRING,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Empty Response")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_empty_ground_truth(self):
        """Test with empty ground truth string."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.EMPTY_STRING,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_empty_query(self):
        """Test with empty query string."""
        results = self._run_evaluation(
            query=self.EMPTY_STRING,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Empty Query")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_single_word_inputs(self):
        """Test with single word inputs."""
        results = self._run_evaluation(
            query="Word?",
            response=self.SINGLE_WORD,
            ground_truth=self.SINGLE_WORD,
        )
        result_data = self._extract_and_print_result(results, "Single Word")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_long_text_inputs(self):
        """Test with very long text inputs."""
        results = self._run_evaluation(
            query="Describe something in detail.",
            response=self.LONG_TEXT,
            ground_truth=self.LONG_TEXT,
        )
        result_data = self._extract_and_print_result(results, "Long Text")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_unicode_text(self):
        """Test with Unicode characters."""
        results = self._run_evaluation(
            query="Translate to Japanese.",
            response=self.UNICODE_TEXT,
            ground_truth=self.UNICODE_TEXT,
        )
        result_data = self._extract_and_print_result(results, "Unicode Text")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_special_characters(self):
        """Test with special characters and punctuation."""
        results = self._run_evaluation(
            query="How are you?",
            response=self.SPECIAL_CHARS,
            ground_truth=self.SPECIAL_CHARS,
        )
        result_data = self._extract_and_print_result(results, "Special Characters")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

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
        """Test with numeric query type (int) - evaluator converts to string."""
        results = self._run_evaluation(
            query=12345,
            response=self.PERFECT_RESPONSE,
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "Numeric Query Type")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_list_response_type(self):
        """Test with list response type - evaluator converts to string."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=["hello", "world"],
            ground_truth=self.PERFECT_GROUND_TRUTH,
        )
        result_data = self._extract_and_print_result(results, "List Response Type")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

    def test_dict_ground_truth_type(self):
        """Test with dict ground truth type - evaluator converts to string."""
        results = self._run_evaluation(
            query=self.VALID_QUERY,
            response=self.PERFECT_RESPONSE,
            ground_truth={"text": "hello"},
        )
        result_data = self._extract_and_print_result(results, "Dict Ground Truth Type")
        self.assert_score_in_range(result_data, min_score=1, max_score=5)

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
        """Test that semantically equivalent responses get high scores."""
        results = self._run_evaluation(
            query="What is H2O?",
            response="H2O is the chemical formula for water.",
            ground_truth="Water has the chemical formula H2O.",
        )
        result_data = self._extract_and_print_result(results, "Semantic Equivalence")
        self.assert_pass_or_fail(result_data)

    # ==================== DOMAIN-SPECIFIC TESTS ====================

    def test_technical_domain_similarity(self):
        """Test with technical/scientific content."""
        results = self._run_evaluation(
            query="Explain DNA replication.",
            response="DNA replication is the process of copying genetic material.",
            ground_truth="DNA replication copies the genetic information stored in DNA.",
        )
        result_data = self._extract_and_print_result(results, "Technical Domain")
        self.assert_pass_or_fail(result_data)

    def test_code_like_content(self):
        """Test with code-like content."""
        results = self._run_evaluation(
            query="Write a hello world function.",
            response="def hello(): print('Hello')",
            ground_truth="def hello_world(): print('Hello, World!')",
        )
        result_data = self._extract_and_print_result(results, "Code Content")
        self.assert_pass_or_fail(result_data)
