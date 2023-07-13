# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift compute metrics component component."""

from data_drift_compute_metrics.io_utils import init_spark
from data_drift_compute_metrics.numerical_data_drift_metrics import compute_numerical_data_drift_measures_tests
import pandas as pd
import pytest
import numpy as np
import unittest


@pytest.mark.unit
class TestComputeDataDriftMetrics(unittest.TestCase):
    """Test class for data drift compute metrics component component and utilities."""

    def test_compute_numerical_data_drift_metrics_normalized_wasserstein_distance(self):
        """Test compute normalized wasserstein distance for numerical metrics."""
        n_obs = 100_000
        mean = 50
        std_dev = 15
        column_values = ['column1']
        numerical_threshold = 100

        x = np.random.normal(mean, std_dev, n_obs)
        y = np.random.normal(mean, std_dev, n_obs)

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

        self.assertAlmostEqual(0, output_df['column1'], 5)
