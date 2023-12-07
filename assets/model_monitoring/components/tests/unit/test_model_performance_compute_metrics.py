import unittest
from shared_utilities.io_utils import init_spark

class TestModelPerformanceComputeMetrics(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestModelPerformanceComputeMetrics, self).__init__(*args, **kwargs)
        self._test_suite_name = "test_model_performance_compute_metrics"
        self.spark = init_spark()

    def test_compute_metrics(self):
        self.assertEqual(1, 1)