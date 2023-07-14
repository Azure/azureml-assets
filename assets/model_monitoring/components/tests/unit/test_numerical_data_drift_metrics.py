# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift compute metrics component component."""

from data_drift_compute_metrics.io_utils import init_spark
from data_drift_compute_metrics.numerical_data_drift_metrics import compute_numerical_data_drift_measures_tests
import pandas as pd
import pytest
import numpy as np
import unittest

test_cases = [
                [100_000, 100_000, 50, 50, 0],
                [100_000, 10_000, 50, 50, 0.19963],
                [100_000, 100_000, 50, 100, 3.3238],
                [100_000, 100_000, 50, 250, 13.3184],
             ]


@pytest.mark.unit
class TestComputeDataDriftMetrics(unittest.TestCase):
    """Test class for data drift compute metrics component component and utilities."""

    def test_compute_numerical_data_drift_metrics_normalized_wasserstein_distance(self):
        """Test compute normalized wasserstein distance for numerical metrics."""
        for x_n_obs, y_n_obs, x_mean, y_mean, expected in test_cases:
            std_dev = 15
            column_values = ['column1']
            numerical_threshold = 100

            x = np.random.normal(x_mean, std_dev, x_n_obs)
            y = np.random.normal(y_mean, std_dev, y_n_obs)

            x_pd_df = pd.DataFrame(data=x, columns=column_values)
            y_pd_df = pd.DataFrame(data=y, columns=column_values)

            spark = init_spark()

            x_df = spark.createDataFrame(x_pd_df)
            y_df = spark.createDataFrame(y_pd_df)

            output_df = compute_numerical_data_drift_measures_tests(
                x_df,
                y_df,
                x_df.count(),
                y_df.count(),
                "NormalizedWassersteinDistance",
                column_values,
                numerical_threshold)
            self.assertAlmostEqual(float(expected), output_df['column1'], 4)
