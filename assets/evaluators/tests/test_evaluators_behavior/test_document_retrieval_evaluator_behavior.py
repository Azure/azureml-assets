# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Document Retrieval Evaluator."""

import pytest
from typing import Any, Dict, List, override
from azure.ai.evaluation._exceptions import EvaluationException
from ..common.base_code_evaluator_runner import BaseCodeEvaluatorRunner
from ...builtin.document_retrieval.evaluator._document_retrieval import DocumentRetrievalEvaluator


@pytest.mark.unittest
class TestDocumentRetrievalEvaluatorBehavior(BaseCodeEvaluatorRunner):
    """
    Behavioral tests for Document Retrieval Evaluator.

    Tests different input formats, thresholds, metrics, and edge cases.
    """

    evaluator_type = DocumentRetrievalEvaluator
    result_key = "ndcg3"  # Primary metric for assertions
    constructor_arg_names = ["ground_truth_label_min", "ground_truth_label_max", "ndcg_threshold",
                             "xdcg_threshold", "fidelity_threshold", "top1_relevance_threshold",
                             "top3_max_relevance_threshold"]

    @property
    def expected_result_fields(self) -> List[str]:
        """Get the expected result fields for document retrieval evaluator."""
        return [
            "ndcg@3", "xdcg@3", "fidelity", "top1_relevance",
            "top3_max_relevance", "holes", "holes_ratio",
            "total_retrieved_documents", "total_ground_truth_documents"
        ]

    # region Test Data
    # Perfect retrieval scenario - top 3 documents match ideal ranking
    PERFECT_GROUND_TRUTH: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 4},
        {"document_id": "doc2", "query_relevance_label": 3},
        {"document_id": "doc3", "query_relevance_label": 2},
        {"document_id": "doc4", "query_relevance_label": 1},
        {"document_id": "doc5", "query_relevance_label": 0},
    ]

    PERFECT_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": "doc1", "relevance_score": 0.95},
        {"document_id": "doc2", "relevance_score": 0.85},
        {"document_id": "doc3", "relevance_score": 0.75},
        {"document_id": "doc4", "relevance_score": 0.65},
        {"document_id": "doc5", "relevance_score": 0.55},
    ]

    # Suboptimal retrieval - documents are in wrong order
    SUBOPTIMAL_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": "doc5", "relevance_score": 0.95},  # Irrelevant doc ranked first
        {"document_id": "doc4", "relevance_score": 0.85},
        {"document_id": "doc3", "relevance_score": 0.75},
        {"document_id": "doc2", "relevance_score": 0.65},
        {"document_id": "doc1", "relevance_score": 0.55},  # Best doc ranked last
    ]

    # Partial match - some retrieved docs have no ground truth (holes)
    PARTIAL_GROUND_TRUTH: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 4},
        {"document_id": "doc2", "query_relevance_label": 3},
    ]

    PARTIAL_RETRIEVED_WITH_HOLES: List[Dict[str, Any]] = [
        {"document_id": "doc1", "relevance_score": 0.95},
        {"document_id": "unknown1", "relevance_score": 0.85},  # Hole
        {"document_id": "doc2", "relevance_score": 0.75},
        {"document_id": "unknown2", "relevance_score": 0.65},  # Hole
    ]

    # All holes scenario - none of the retrieved docs have ground truth
    ALL_HOLES_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": "unknown1", "relevance_score": 0.95},
        {"document_id": "unknown2", "relevance_score": 0.85},
        {"document_id": "unknown3", "relevance_score": 0.75},
    ]

    # Empty retrieved documents
    EMPTY_RETRIEVED: List[Dict[str, Any]] = []

    # Single document scenarios
    SINGLE_GROUND_TRUTH: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 4},
    ]

    SINGLE_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": "doc1", "relevance_score": 0.95},
    ]

    # Large dataset (within limits)
    LARGE_GROUND_TRUTH: List[Dict[str, Any]] = [
        {"document_id": f"doc{i}", "query_relevance_label": i % 5} for i in range(100)
    ]

    LARGE_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": f"doc{i}", "relevance_score": 1.0 - (i * 0.01)} for i in range(100)
    ]

    # Edge case with same relevance scores
    SAME_SCORE_RETRIEVED: List[Dict[str, Any]] = [
        {"document_id": "doc1", "relevance_score": 0.5},
        {"document_id": "doc2", "relevance_score": 0.5},
        {"document_id": "doc3", "relevance_score": 0.5},
    ]

    # Custom label range ground truth (0-10 scale)
    CUSTOM_RANGE_GROUND_TRUTH: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 10},
        {"document_id": "doc2", "query_relevance_label": 7},
        {"document_id": "doc3", "query_relevance_label": 5},
        {"document_id": "doc4", "query_relevance_label": 2},
        {"document_id": "doc5", "query_relevance_label": 0},
    ]

    # Invalid data scenarios
    INVALID_GROUND_TRUTH_MISSING_ID: List[Dict[str, Any]] = [
        {"query_relevance_label": 4},  # Missing document_id
    ]

    INVALID_GROUND_TRUTH_MISSING_LABEL: List[Dict[str, Any]] = [
        {"document_id": "doc1"},  # Missing query_relevance_label
    ]

    INVALID_GROUND_TRUTH_STRING_LABEL: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": "high"},  # String instead of int
    ]

    INVALID_GROUND_TRUTH_FLOAT_LABEL: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 3.5},  # Float instead of int
    ]

    INVALID_GROUND_TRUTH_OUT_OF_RANGE_HIGH: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": 5},  # Above default max of 4
    ]

    INVALID_GROUND_TRUTH_OUT_OF_RANGE_LOW: List[Dict[str, Any]] = [
        {"document_id": "doc1", "query_relevance_label": -1},  # Below default min of 0
    ]

    INVALID_RETRIEVED_MISSING_ID: List[Dict[str, Any]] = [
        {"relevance_score": 0.95},  # Missing document_id
    ]

    INVALID_RETRIEVED_MISSING_SCORE: List[Dict[str, Any]] = [
        {"document_id": "doc1"},  # Missing relevance_score
    ]

    INVALID_RETRIEVED_STRING_SCORE: List[Dict[str, Any]] = [
        {"document_id": "doc1", "relevance_score": "high"},  # String instead of float
    ]
    # endregion

    @override
    def _extract_and_print_result(self, results: Dict[str, Any], test_label: str) -> Dict[str, Any]:
        """Extract and print detailed result metrics for Document Retrieval Evaluator.

        Args:
            results: Dictionary containing evaluation results.
            test_label: Descriptive label for the test (printed in output).
        """
        result = super()._extract_and_print_result(results, test_label)

        # Document Retrieval Evaluator specific fields
        ndcg = results.get("ndcg@3")
        xdcg = results.get("xdcg@3")
        fidelity = results.get("fidelity")
        top1_relevance = results.get("top1_relevance")
        top3_max_relevance = results.get("top3_max_relevance")
        holes = results.get("holes")
        holes_ratio = results.get("holes_ratio")
        total_retrieved = results.get("total_retrieved_documents")
        total_ground_truth = results.get("total_ground_truth_documents")
        if ndcg is not None:
            print(f"  NDCG@3: {ndcg}")
            result["ndcg3"] = ndcg
        if xdcg is not None:
            print(f"  XDCG@3: {xdcg}")
            result["xdcg3"] = xdcg
        if fidelity is not None:
            print(f"  Fidelity: {fidelity}")
            result["fidelity"] = fidelity
        if top1_relevance is not None:
            print(f"  Top1 Relevance: {top1_relevance}")
            result["top1_relevance"] = top1_relevance
        if top3_max_relevance is not None:
            print(f"  Top3 Max Relevance: {top3_max_relevance}")
            result["top3_max_relevance"] = top3_max_relevance
        if holes is not None:
            print(f"  Holes: {holes}")
            result["holes"] = holes
        if holes_ratio is not None:
            print(f"  Holes Ratio: {holes_ratio}")
            result["holes_ratio"] = holes_ratio
        if total_retrieved is not None:
            print(f"  Total Retrieved Documents: {total_retrieved}")
            result["total_retrieved"] = total_retrieved
        if total_ground_truth is not None:
            print(f"  Total Ground Truth Documents: {total_ground_truth}")
            result["total_ground_truth"] = total_ground_truth
        return result

    def assert_valid_metrics(self, result_data: Dict[str, Any]):
        """Assert that all metrics are present and valid."""
        assert result_data["ndcg3"] is not None
        assert result_data["xdcg3"] is not None
        assert result_data["fidelity"] is not None
        assert result_data["top1_relevance"] is not None
        assert result_data["top3_max_relevance"] is not None
        assert result_data["holes"] is not None
        assert result_data["holes_ratio"] is not None
        assert result_data["total_retrieved"] is not None
        assert result_data["total_ground_truth"] is not None

    # ==================== PERFECT RETRIEVAL TESTS ====================

    def test_perfect_retrieval(self):
        """Test with perfect document retrieval matching ideal ranking."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Perfect Retrieval")
        self.assert_valid_metrics(result_data)
        # Perfect retrieval should have NDCG = 1.0
        assert result_data["ndcg3"] == 1.0
        # No holes expected
        assert result_data["holes"] == 0
        assert result_data["holes_ratio"] == 0.0
        # Top document should have highest relevance (4)
        assert result_data["top1_relevance"] == 4
        assert result_data["top3_max_relevance"] == 4

    def test_perfect_retrieval_fidelity(self):
        """Test fidelity score with perfect retrieval."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Perfect Retrieval Fidelity")
        # Fidelity should be 1.0 for perfect retrieval
        assert result_data["fidelity"] == 1.0

    # ==================== SUBOPTIMAL RETRIEVAL TESTS ====================

    def test_suboptimal_retrieval(self):
        """Test with suboptimal document retrieval (reversed order)."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.SUBOPTIMAL_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Suboptimal Retrieval")
        self.assert_valid_metrics(result_data)
        # NDCG should be less than 1.0 for suboptimal ordering
        assert result_data["ndcg3"] < 1.0
        # Top document has lowest relevance (0) since order is reversed
        assert result_data["top1_relevance"] == 0
        # Still no holes since all docs are in ground truth
        assert result_data["holes"] == 0

    def test_suboptimal_xdcg(self):
        """Test XDCG score with suboptimal retrieval."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.SUBOPTIMAL_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Suboptimal XDCG")
        # XDCG should be lower for poorly ranked results
        # Compare with perfect retrieval
        perfect_results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        assert result_data["xdcg3"] < perfect_results["xdcg@3"]

    # ==================== HOLES TESTS ====================

    def test_partial_retrieval_with_holes(self):
        """Test retrieval with documents not in ground truth (holes)."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PARTIAL_GROUND_TRUTH,
            retrieved_documents=self.PARTIAL_RETRIEVED_WITH_HOLES,
        )
        result_data = self._extract_and_print_result(results, "Partial Retrieval with Holes")
        self.assert_valid_metrics(result_data)
        # Should have 2 holes (unknown1 and unknown2)
        assert result_data["holes"] == 2
        assert result_data["holes_ratio"] == 0.5  # 2 out of 4 documents

    def test_all_holes_retrieval(self):
        """Test retrieval where no documents have ground truth (all holes)."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PARTIAL_GROUND_TRUTH,
            retrieved_documents=self.ALL_HOLES_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "All Holes Retrieval")
        self.assert_valid_metrics(result_data)
        # All 3 documents are holes
        assert result_data["holes"] == 3
        assert result_data["holes_ratio"] == 1.0
        # Metrics should be zero since no docs have labels
        assert result_data["ndcg3"] == 0
        assert result_data["xdcg3"] == 0
        assert result_data["fidelity"] == 0

    def test_zero_holes(self):
        """Test perfect retrieval has zero holes."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        # Verify holes result passes with threshold of 0
        assert results.get("holes_result") == "pass"
        assert results.get("holes_ratio_result") == "pass"

    # ==================== EMPTY RETRIEVED DOCUMENTS TESTS ====================

    def test_empty_retrieved_documents(self):
        """Test with empty retrieved documents list."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.EMPTY_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Empty Retrieved Documents")
        self.assert_valid_metrics(result_data)
        # All metrics should be zero
        assert result_data["ndcg3"] == 0.0
        assert result_data["xdcg3"] == 0.0
        assert result_data["fidelity"] == 0.0
        assert result_data["top1_relevance"] == 0.0
        assert result_data["top3_max_relevance"] == 0.0
        assert result_data["holes"] == 0
        assert result_data["total_retrieved"] == 0

    # ==================== SINGLE DOCUMENT TESTS ====================

    def test_single_document_retrieval(self):
        """Test with single document in both ground truth and retrieval."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.SINGLE_GROUND_TRUTH,
            retrieved_documents=self.SINGLE_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Single Document Retrieval")
        self.assert_valid_metrics(result_data)
        assert result_data["total_retrieved"] == 1
        assert result_data["total_ground_truth"] == 1
        assert result_data["top1_relevance"] == 4
        assert result_data["holes"] == 0

    # ==================== CUSTOM LABEL RANGE TESTS ====================

    def test_custom_label_range(self):
        """Test with custom ground truth label range (0-10)."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.CUSTOM_RANGE_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
            ground_truth_label_min=0,
            ground_truth_label_max=10,
        )
        result_data = self._extract_and_print_result(results, "Custom Label Range (0-10)")
        self.assert_valid_metrics(result_data)
        # Top document should have relevance of 10
        assert result_data["top1_relevance"] == 10

    def test_custom_label_range_with_defaults(self):
        """Test that custom labels outside default range raise error with default settings."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.CUSTOM_RANGE_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
            # Using default label range 0-4, but data has labels up to 10
        )
        result_data = self._extract_and_print_result(results, "Custom Labels with Default Range")
        self.assert_error(result_data)

    # ==================== SAME RELEVANCE SCORE TESTS ====================

    def test_same_relevance_scores(self):
        """Test with retrieved documents having same relevance scores."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.SAME_SCORE_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Same Relevance Scores")
        self.assert_valid_metrics(result_data)
        assert result_data["total_retrieved"] == 3

    # ==================== LARGE DATASET TESTS ====================

    def test_large_dataset(self):
        """Test with larger dataset (100 documents)."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.LARGE_GROUND_TRUTH,
            retrieved_documents=self.LARGE_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Large Dataset")
        self.assert_valid_metrics(result_data)
        assert result_data["total_retrieved"] == 100
        assert result_data["total_ground_truth"] == 100

    # ==================== THRESHOLD TESTS ====================

    def test_ndcg_threshold_pass(self):
        """Test NDCG passes with low threshold."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
            ndcg_threshold=0.5,
        )
        assert results.get("ndcg@3_result") == "pass"

    def test_ndcg_threshold_fail(self):
        """Test NDCG fails with high threshold on suboptimal retrieval."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.SUBOPTIMAL_RETRIEVED,
            ndcg_threshold=0.9,
        )
        # Suboptimal retrieval should fail with high threshold
        assert results.get("ndcg@3_result") == "fail"

    def test_fidelity_threshold(self):
        """Test fidelity threshold evaluation."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
            fidelity_threshold=0.9,
        )
        # Perfect retrieval should pass fidelity threshold
        assert results.get("fidelity_result") == "pass"

    def test_all_thresholds_present(self):
        """Test that all threshold-related keys are present in output."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        # Check threshold values are in output
        assert "ndcg@3_threshold" in results
        assert "xdcg@3_threshold" in results
        assert "fidelity_threshold" in results
        assert "top1_relevance_threshold" in results
        assert "top3_max_relevance_threshold" in results
        assert "holes_threshold" in results
        assert "holes_ratio_threshold" in results

    # ==================== ERROR HANDLING TESTS ====================

    def test_empty_ground_truth(self):
        """Test with empty ground truth list."""
        results = self._run_evaluation(
            retrieval_ground_truth=[],
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Empty Ground Truth")
        self.assert_error(result_data)

    def test_none_ground_truth(self):
        """Test with None ground truth."""
        results = self._run_evaluation(
            retrieval_ground_truth=None,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "None Ground Truth")
        self.assert_error(result_data)

    def test_invalid_ground_truth_missing_document_id(self):
        """Test with ground truth missing document_id."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_MISSING_ID,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Missing Document ID in Ground Truth")
        self.assert_error(result_data)

    def test_invalid_ground_truth_missing_label(self):
        """Test with ground truth missing query_relevance_label."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_MISSING_LABEL,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Missing Label in Ground Truth")
        self.assert_error(result_data)

    def test_invalid_ground_truth_string_label(self):
        """Test with ground truth having string label instead of int."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_STRING_LABEL,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "String Label in Ground Truth")
        self.assert_error(result_data)

    def test_invalid_ground_truth_float_label(self):
        """Test with ground truth having float label instead of int."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_FLOAT_LABEL,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Float Label in Ground Truth")
        self.assert_error(result_data)

    def test_invalid_ground_truth_label_above_max(self):
        """Test with ground truth label above configured maximum."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_OUT_OF_RANGE_HIGH,
            retrieved_documents=self.SINGLE_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Label Above Max")
        self.assert_error(result_data)

    def test_invalid_ground_truth_label_below_min(self):
        """Test with ground truth label below configured minimum."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.INVALID_GROUND_TRUTH_OUT_OF_RANGE_LOW,
            retrieved_documents=self.SINGLE_RETRIEVED,
        )
        result_data = self._extract_and_print_result(results, "Label Below Min")
        self.assert_error(result_data)

    def test_invalid_retrieved_missing_document_id(self):
        """Test with retrieved documents missing document_id."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.SINGLE_GROUND_TRUTH,
            retrieved_documents=self.INVALID_RETRIEVED_MISSING_ID,
        )
        result_data = self._extract_and_print_result(results, "Missing Document ID in Retrieved")
        self.assert_error(result_data)

    def test_invalid_retrieved_missing_score(self):
        """Test with retrieved documents missing relevance_score."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.SINGLE_GROUND_TRUTH,
            retrieved_documents=self.INVALID_RETRIEVED_MISSING_SCORE,
        )
        result_data = self._extract_and_print_result(results, "Missing Score in Retrieved")
        self.assert_error(result_data)

    def test_invalid_retrieved_string_score(self):
        """Test with retrieved documents having string score instead of float."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.SINGLE_GROUND_TRUTH,
            retrieved_documents=self.INVALID_RETRIEVED_STRING_SCORE,
        )
        result_data = self._extract_and_print_result(results, "String Score in Retrieved")
        self.assert_error(result_data)

    def test_invalid_label_range_min_equals_max(self):
        """Test with min label equal to max label."""
        with pytest.raises(EvaluationException):
            self._init_evaluator(ground_truth_label_min=2, ground_truth_label_max=2)

    def test_invalid_label_range_min_greater_than_max(self):
        """Test with min label greater than max label."""
        with pytest.raises(EvaluationException):
            self._init_evaluator(ground_truth_label_min=5, ground_truth_label_max=2)

    def test_invalid_label_min_not_integer(self):
        """Test with non-integer min label."""
        with pytest.raises(EvaluationException):
            self._init_evaluator(ground_truth_label_min=0.5, ground_truth_label_max=4)

    def test_invalid_label_max_not_integer(self):
        """Test with non-integer max label."""
        with pytest.raises(EvaluationException):
            self._init_evaluator(ground_truth_label_min=0, ground_truth_label_max=4.5)

    # ==================== OUTPUT STRUCTURE TESTS ====================

    def test_output_contains_all_metrics(self):
        """Test that output contains all expected metrics."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        # Core metrics
        assert "ndcg@3" in results
        assert "xdcg@3" in results
        assert "fidelity" in results
        assert "top1_relevance" in results
        assert "top3_max_relevance" in results
        assert "holes" in results
        assert "holes_ratio" in results
        assert "total_retrieved_documents" in results
        assert "total_ground_truth_documents" in results

    def test_output_contains_result_keys(self):
        """Test that output contains pass/fail result keys."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        # Result keys
        assert "ndcg@3_result" in results
        assert "xdcg@3_result" in results
        assert "fidelity_result" in results
        assert "top1_relevance_result" in results
        assert "top3_max_relevance_result" in results
        assert "holes_result" in results
        assert "holes_ratio_result" in results

    def test_output_contains_higher_is_better_keys(self):
        """Test that output contains higher_is_better indicator keys."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        # Higher is better for main metrics
        assert results.get("ndcg@3_higher_is_better") is True
        assert results.get("xdcg@3_higher_is_better") is True
        assert results.get("fidelity_higher_is_better") is True
        # Lower is better for holes
        assert results.get("holes_higher_is_better") is False
        assert results.get("holes_ratio_higher_is_better") is False

    def test_result_values_are_pass_or_fail(self):
        """Test that result values are either 'pass' or 'fail'."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        result_keys = [k for k in results.keys() if k.endswith("_result")]
        for key in result_keys:
            assert results[key] in ["pass", "fail"], f"{key} should be 'pass' or 'fail'"

    # ==================== METRICS RANGE TESTS ====================

    def test_ndcg_range(self):
        """Test that NDCG is within valid range [0, 1]."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        assert 0.0 <= results["ndcg@3"] <= 1.0

        results_suboptimal = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.SUBOPTIMAL_RETRIEVED,
        )
        assert 0.0 <= results_suboptimal["ndcg@3"] <= 1.0

    def test_holes_ratio_range(self):
        """Test that holes_ratio is within valid range [0, 1]."""
        # No holes
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        assert 0.0 <= results["holes_ratio"] <= 1.0

        # All holes
        results_holes = self._run_evaluation(
            retrieval_ground_truth=self.PARTIAL_GROUND_TRUTH,
            retrieved_documents=self.ALL_HOLES_RETRIEVED,
        )
        assert 0.0 <= results_holes["holes_ratio"] <= 1.0

    def test_fidelity_range(self):
        """Test that fidelity is within valid range [0, 1]."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        assert 0.0 <= results["fidelity"] <= 1.0

    # ==================== DOCUMENT COUNT TESTS ====================

    def test_total_document_counts(self):
        """Test that total document counts are correct."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=self.PERFECT_RETRIEVED,
        )
        assert results["total_retrieved_documents"] == len(self.PERFECT_RETRIEVED)
        assert results["total_ground_truth_documents"] == len(self.PERFECT_GROUND_TRUTH)

    def test_partial_retrieval_counts(self):
        """Test document counts with partial retrieval."""
        results = self._run_evaluation(
            retrieval_ground_truth=self.PARTIAL_GROUND_TRUTH,
            retrieved_documents=self.PARTIAL_RETRIEVED_WITH_HOLES,
        )
        assert results["total_retrieved_documents"] == len(self.PARTIAL_RETRIEVED_WITH_HOLES)
        assert results["total_ground_truth_documents"] == len(self.PARTIAL_GROUND_TRUTH)

    # ==================== INTEGER RELEVANCE SCORE TESTS ====================

    def test_integer_relevance_scores_accepted(self):
        """Test that integer relevance scores are accepted."""
        retrieved_with_int_scores = [
            {"document_id": "doc1", "relevance_score": 95},  # int instead of float
            {"document_id": "doc2", "relevance_score": 85},
            {"document_id": "doc3", "relevance_score": 75},
        ]
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=retrieved_with_int_scores,
        )
        result_data = self._extract_and_print_result(results, "Integer Relevance Scores")
        self.assert_valid_metrics(result_data)

    # ==================== DUPLICATE DOCUMENT TESTS ====================

    def test_duplicate_documents_in_retrieval(self):
        """Test behavior with duplicate document IDs in retrieved documents."""
        retrieved_with_duplicates = [
            {"document_id": "doc1", "relevance_score": 0.95},
            {"document_id": "doc1", "relevance_score": 0.85},  # Duplicate
            {"document_id": "doc2", "relevance_score": 0.75},
        ]
        results = self._run_evaluation(
            retrieval_ground_truth=self.PERFECT_GROUND_TRUTH,
            retrieved_documents=retrieved_with_duplicates,
        )
        result_data = self._extract_and_print_result(results, "Duplicate Documents in Retrieval")
        # Should process without error (behavior may vary based on implementation)
        assert result_data["total_retrieved"] == 3
