# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Model Performance Compute Metrics component."""

import pytest
import unittest

from shared_utilities.io_utils import init_spark
from model_performance_metrics.data_reader import DataReaderFactory, BaseTaskReader, MetricsDTO
from model_performance_metrics.compute_metrics import EvaluatorFactory, ClassifierEvaluator, RegressorEvaluator

test_data_classification = [{
    "ground_truth": {
        "id": [1, 2, 3],
        "label": [0, 1, 0]
    },
    "predictions": {
        "id": [1, 2, 3],
        "label": [0, 0, 1]
    },
    "result": {'accuracy': 0.3333333333333333, 'precision_score_macro': 0.25, 'recall_score_macro': 0.25}
}]

test_data_regression = [{
    "ground_truth": {
        "id": [1, 2, 3],
        "label": [0.1, 0.2, 0.3]
    },
    "predictions": {
        "id": [1, 2, 3],
        "label": [0.1, 0.1, 0.4]
    },
    "result": {'root_mean_squared_error': 0.08164965809277262, 'mean_absolute_error': 0.06666666666666668}
}]


@pytest.mark.unit1
class TestModelPerformanceComputeMetrics(unittest.TestCase):
    """Unit test class for model performance compute metrics."""

    def __init__(self, *args, **kwargs):
        """Initialize the unit test class."""
        super(TestModelPerformanceComputeMetrics, self).__init__(*args, **kwargs)
        self._test_suite_name = "test_model_performance_compute_metrics"
        self.spark = init_spark()

    def get_test_cases_with_res(self, task_type):
        """Get test cases with expected results."""
        if task_type == "tabular-classification":
            for test_case in test_data_classification:
                yield MetricsDTO(test_case["ground_truth"]["label"], test_case["predictions"]["label"]), \
                      test_case["result"]
        elif task_type == "tabular-regression":
            for test_case in test_data_regression:
                yield MetricsDTO(test_case["ground_truth"]["label"], test_case["predictions"]["label"]), \
                      test_case["result"]
        else:
            raise Exception("Invalid task type")

    def test_data_reader_factory(self):
        """Test data reader factory."""
        assert isinstance(DataReaderFactory().get_reader("tabular-classification"), BaseTaskReader)
        assert isinstance(DataReaderFactory().get_reader("tabular-regression"), BaseTaskReader)

    def test_evaluator_factory(self):
        """Test evaluator factory."""
        assert isinstance(EvaluatorFactory().get_evaluator("tabular-classification"), ClassifierEvaluator)
        assert isinstance(EvaluatorFactory().get_evaluator("tabular-regression"), RegressorEvaluator)

    def test_classifier_evaluator(self):
        """Test classifier evaluator."""
        classifier_evaluator = EvaluatorFactory().get_evaluator("tabular-classification")
        for metrics_dto, expected_result in self.get_test_cases_with_res("tabular-classification"):
            metrics = classifier_evaluator.evaluate(metrics_dto)
            for key in expected_result.keys():
                self.assertAlmostEqual(metrics["metrics"][key], expected_result[key])

    def test_regressor_evaluator(self):
        """Test regressor evaluator."""
        regressor_evaluator = EvaluatorFactory().get_evaluator("tabular-regression")
        for metrics_dto, expected_result in self.get_test_cases_with_res("tabular-regression"):
            metrics = regressor_evaluator.evaluate(metrics_dto)
            for key in expected_result.keys():
                self.assertAlmostEqual(metrics["metrics"][key], expected_result[key])
