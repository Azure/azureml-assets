# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This file contains unit tests for the Categorical Data Drift Metrics.

It compares the drift measures of metrics implemented in Spark against their SciPy implementation
"""

from data_drift_compute_metrics.categorical_data_drift_metrics import compute_categorical_data_drift_measures_tests
from shared_utilities.io_utils import init_spark
from shared_utilities.constants import(
    JENSEN_SHANNON_DISTANCE_METRIC_NAME,
    PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
    POPULATION_STABILITY_INDEX_METRIC_NAME
)
from scipy.spatial import distance
from scipy.stats import chisquare
import pandas as pd
import pyspark.sql as pyspark_sql
import pytest
import numpy as np
import unittest
import math
import random

distance_measures = [
    JENSEN_SHANNON_DISTANCE_METRIC_NAME,
    POPULATION_STABILITY_INDEX_METRIC_NAME,
    PEARSONS_CHI_SQUARED_TEST_METRIC_NAME,
]


test_cases = [
    {
     "name": "1a",
     "scenario": "Distance between identical distributions should be exactly 0.",
     "baseline_col": {'a': 97, 'b': 105, 'c': 99, 'd': 98, 'e': 101, 'f': 102, 'g': 97, 'h': 103},
     "production_col": {'a': 97, 'b': 105, 'c': 99, 'd': 98, 'e': 101, 'f': 102, 'g': 97, 'h': 103},
    },
    {
     "name": "1b",
     "scenario": "Minimal drift. Both distributions contain the same categories.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 97, 'b': 105, 'c': 99, 'd': 98, 'e': 101, 'f': 102, 'g': 97, 'h': 103},
    },
    {
     "name": "1c",
     "scenario": "Marginal drift. Both distributions contain the same categories.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 100, 'b': 110, 'c': 90, 'd': 100, 'e': 100, 'f': 100, 'g': 110, 'h': 100},
    },
    {
     "name": "1d",
     "scenario": "Marginal drift with different sample size. Metric value should be very similar to test 1c.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 10, 'b': 11, 'c': 9, 'd': 10, 'e': 10, 'f': 10, 'g': 11, 'h': 10},
    },
    {
     "name": "1e",
     "scenario": "Drifted distributions. Both distributions contain the same categories.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 180, 'b': 80, 'c': 70, 'd': 170, 'e': 200, 'f': 10, 'g': 130, 'h': 100},
    },
    {
     "name": "1f",
     "scenario": "Drift caused by a single category. Both distributions contain the same categories.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 1},
    },
    {
     "name": "1g",
     "scenario": "Drifted distributions. Production dataset is missing some categories.",
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'a': 100, 'b': 100, 'c': 100, 'f': 100, 'g': 100, 'h': 1},
    },
    {
     "name": "1h",
     "scenario": (
         "Distributions with no categories in common. "
         "Bound distance measures (Jensen-Shannon) should reach their maximum theoretical value."),
     "baseline_col": {'a': 100, 'b': 100, 'c': 100, 'd': 100, 'e': 100, 'f': 100, 'g': 100, 'h': 100},
     "production_col": {'i': 1, 'j': 1, 'k': 2},
    },
    {
     "name": "2a",
     "scenario": "Dual value identical distributions with the same sample size. Distance measures should be 0.",
     "baseline_col": {'a': 100, 'b': 100},
     "production_col": {'a': 100, 'b': 100},
    },
    {
     "name": "2b",
     "scenario": "Dual value identical distributions with different sample sizes. Distance measures should be 0.",
     "baseline_col": {'a': 100, 'b': 100},
     "production_col": {'a': 34, 'b': 34},
    },
    {
     "name": "2c",
     "scenario": "Drifted Dual value distributions.",
     "baseline_col": {'a': 100, 'b': 100},
     "production_col": {'a': 92, 'b': 23},
    },
    {
     "name": "2d",
     "scenario": (
         "Dual value distributions with different values. "
         "Bound distance measures (Jensen-Shannon) should reach their maximum theoretical value."),
     "baseline_col": {'a': 100, 'b': 100},
     "production_col": {'c': 100, 'd': 100},
    },
    {
     "name": "3a",
     "scenario": "Distributions with low sample sizes.",
     "baseline_col": {'a': 3, 'b': 1, 'c': 4},
     "production_col": {'a': 2, 'b': 3},
    },
]


@pytest.mark.unit
class TestComputeCategoricalDataDriftMetrics(unittest.TestCase):
    """Test class for categorical data drift compute metrics component and utilities."""

    def __init__(self, *args, **kwargs):
        """Initialize TestComputeCategoricalDataDriftMetrics and initialize the Spark session."""
        super(TestComputeCategoricalDataDriftMetrics, self).__init__(*args, **kwargs)
        self.spark = init_spark()

    def round_to_n_significant_digits(self, number, n):
        """
        Round a given number to n significant digits.

        :param number: The number to round.
        :param n: The number of significant digits to round to.
        :return: The rounded number.
        """
        if number == 0:
            return 0
        else:
            return round(number, n - int(math.floor(math.log10(abs(number)))) - 1)

    def get_metric_value(self, df: pyspark_sql.DataFrame, metric_name: str):
        """Get metric value of the first row of a given column from a dataframe."""
        return df.filter(f"metric_name = '{metric_name}'").first().metric_value

    def create_spark_df(self, categorical_data):
        """
        Create a Spark DataFrame from the given categorical data.

        :param categorical_data: A list of categorical data.
        :return: A Spark DataFrame with the categorical data.
        """
        column_values = ['column1']
        pd_df = pd.DataFrame(data=categorical_data, columns=column_values)
        df = self.spark.createDataFrame(pd_df)
        return df

    def create_categorical_data_from_dict(self, sample_dict: dict):
        """
        Create a list of categorical data from the given dictionary.

        :param sample_dict: A dictionary mapping categories to their frequencies.
        :return: A list of categorical data.
        """
        cat_list = []
        for key, value in sample_dict.items():
            for i in range(value):
                cat_list.append(key)
        random.shuffle(cat_list)
        return cat_list

    # # # EXPECTED JENSEN-SHANNON DISTANCE # # #

    def jensen_shannon_distance_categorical(self, x_list, y_list):
        """
        Compute the Jensen-Shannon distance between two lists of categorical data.

        :param x_list: A list of categorical data for the first distribution.
        :param y_list: A list of categorical data for the second distribution.
        :return: The Jensen-Shannon distance between the two distributions.
        """
        # unique values observed in x and y
        values = set(x_list + y_list)

        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])

        x_ratios = x_counts / np.sum(x_counts)  # Optional as JS-D normalizes probability vectors
        y_ratios = y_counts / np.sum(y_counts)

        return distance.jensenshannon(x_ratios, y_ratios, base=2)

    # # # EXPECTED PSI # # #

    def psi_categorical(self, x_list, y_list):
        """
        Compute the Population Stability Index (PSI) between two lists of categorical data.

        :param x_list: A list of categorical data for the first distribution.
        :param y_list: A list of categorical data for the second distribution.
        :return: The PSI between the two distributions.
        """
        # unique values observed in x and y
        values = set(x_list + y_list)

        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])

        # Laplace smoothing (incrementing the count of each bin by 1)
        # to avoid zero values in bins and have the SPI value be finit
        x_counts = x_counts + 1
        y_counts = y_counts + 1

        x_ratios = x_counts / np.sum(x_counts)
        y_ratios = y_counts / np.sum(y_counts)

        psi = 0
        for i in range(len(x_ratios)):
            psi += (y_ratios[i] - x_ratios[i]) * np.log(y_ratios[i] / x_ratios[i])

        return psi

    # # # EXPECTED CHI SQUARED TEST # # #

    def chi_squared_test_categorical(self, x_list, y_list):
        """
        Compute the Pearson's Chi-squared test between two lists of categorical data.

        :param x_list: A list of categorical data for the first distribution.
        :param y_list: A list of categorical data for the second distribution.
        :return: The p-value of the Pearson's Chi-squared test between the two distributions.
        """
        values = set(x_list + y_list)

        x_counts = np.array([x_list.count(value) for value in values])
        y_counts = np.array([y_list.count(value) for value in values])

        x_ratios = x_counts / np.sum(x_counts)
        expected_y_counts = x_ratios * len(y_list)

        return chisquare(y_counts, expected_y_counts).pvalue

    # # # MAIN TEST FUNCTION # # #

    def test_compute_categorical_data_drift_metrics(self):
        """Test compute distance measures for categorical metrics."""
        column_values = ['column1']
        numerical_threshold = 0.1

        for distance_measure in distance_measures:
            print('#########################################################################################')
            print(f'TESTING DISTANCE MEASURE: {distance_measure}')
            print('#########################################################################################')

            for test_case in test_cases:
                x = self.create_categorical_data_from_dict(test_case['baseline_col'])
                y = self.create_categorical_data_from_dict(test_case['production_col'])

                if distance_measure == JENSEN_SHANNON_DISTANCE_METRIC_NAME:
                    expected_distance = self.jensen_shannon_distance_categorical(x, y)
                elif distance_measure == POPULATION_STABILITY_INDEX_METRIC_NAME:
                    expected_distance = self.psi_categorical(x, y)
                elif distance_measure == PEARSONS_CHI_SQUARED_TEST_METRIC_NAME:
                    expected_distance = self.chi_squared_test_categorical(x, y)
                else:
                    raise ValueError(f"Distance measure {distance_measure} not in {distance_measures}")

                x_df = self.create_spark_df(x)
                y_df = self.create_spark_df(y)
                x_count = x_df.count()
                y_count = y_df.count()

                output_df = compute_categorical_data_drift_measures_tests(
                    x_df,
                    y_df,
                    x_count,
                    y_count,
                    distance_measure,
                    column_values,
                    numerical_threshold)

                metric_value = self.get_metric_value(output_df, distance_measure)

                print('-------------------------')
                print(f'test: {test_case["name"]}')
                print(f'test scenario: {test_case["scenario"]}')
                print(f'expected value: {self.round_to_n_significant_digits(expected_distance, 4)}')
                print(f'measured value: {self.round_to_n_significant_digits(metric_value, 4)}')

                if (abs(expected_distance) < 1e-6):
                    self.assertAlmostEqual(metric_value, 0, 6)
                else:
                    self.assertAlmostEqual(float(expected_distance)/metric_value, 1.0, 1)
