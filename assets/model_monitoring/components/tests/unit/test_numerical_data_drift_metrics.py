# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift compute metrics component component."""

from data_drift_compute_metrics.numerical_data_drift_metrics import compute_numerical_data_drift_measures_tests
from shared_utilities.io_utils import init_spark
import pandas as pd
import pyspark.sql as pyspark_sql
import pytest
import numpy as np
import unittest

test_cases = [
                [100_000, 10_000, 50, 50, 0.01],
                [100_000, 100_000, 50, 100, 3.32],
                [100_000, 100_000, 50, 250, 13.32],
             ]


@pytest.mark.unit
class TestComputeDataDriftMetrics(unittest.TestCase):
    """Test class for data drift compute metrics component component and utilities."""

    def get_metric_value(self, df: pyspark_sql.DataFrame, metric_name: str):
        """Get metric value of the first row of a given column from a dataframe."""
        return df.filter(f"metric_name = '{metric_name}'").first().metric_value

    def test_compute_numerical_data_drift_metrics_normalized_wasserstein_distance_identical_distribution(self):
        """Test compute normalized wasserstein distance for numerical metrics when inputs are identitcal."""
        x_n_obs = 100_000
        x_mean = 50
        std_dev = 15
        column_values = ['column1']
        numerical_threshold = 100

        x = np.random.normal(x_mean, std_dev, x_n_obs)

        x_pd_df = pd.DataFrame(data=x, columns=column_values)
        y_pd_df = pd.DataFrame(data=x, columns=column_values)

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

        metric_value = self.get_metric_value(output_df, "NormalizedWassersteinDistance")
        self.assertAlmostEqual(0.0, metric_value, 4)

    def test_compute_numerical_data_drift_metrics_normalized_wasserstein_distance(self):
        """Test compute normalized wasserstein distance for numerical metrics."""
        std_dev = 15
        column_values = ['column1']
        numerical_threshold = 100

        for x_n_obs, y_n_obs, x_mean, y_mean, expected in test_cases:
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

            metric_value = self.get_metric_value(output_df, "NormalizedWassersteinDistance")
            self.assertAlmostEqual(float(expected), metric_value, 1)
