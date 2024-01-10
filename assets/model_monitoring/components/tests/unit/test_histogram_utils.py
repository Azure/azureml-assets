# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the df utilities."""

from pyspark.sql import SparkSession, DataFrame
from src.shared_utilities.df_utils import get_numerical_cols_with_df
from src.shared_utilities.histogram_utils import (
        get_dual_histogram_bin_edges
    )
import math
import pandas as pd
import pytest


@pytest.mark.unit
class TestDFUtils:
    """Test class for histogram utilities."""

    def _num_bins_by_struges_algorithm(self, df: DataFrame) -> int:
        """For testing suite, calculate number of bins for a dataset using struges algorithm."""
        num_bins = math.log2(df.count()) + 1
        return math.ceil(num_bins)

    def test_get_dual_histogram_bin_edges(self):
        """Test with mixed columns expect succeed."""
        column_dtype_map = {
            'col1': 'int',
            'col2': 'float',
            'col3': 'double',
            'col4': 'decimal',
            'col5': 'string'
        }
        baseline_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
            'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
        })
        production_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5, 6, 7],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7],
            'col3': [1.11, 2.22, 3.33, 4.44, 5.55, 6.66, 7.77],
            'col4': [1.111, 2.222, 3.333, 4.444, 5.555, 6.666, 7.777]
        })
        baseline_df = self.init_spark().createDataFrame(baseline_df)
        production_df = self.init_spark().createDataFrame(production_df)
        numerical_columns = get_numerical_cols_with_df(column_dtype_map, baseline_df)

        all_edges = get_dual_histogram_bin_edges(
            baseline_df, production_df, baseline_df.count(), production_df.count(), numerical_columns
        )

        assert all_edges is not None
        for col in numerical_columns:
            assert all_edges.get(col, None) is not None
            assert len(all_edges[col]) == self._num_bins_by_struges_algorithm(baseline_df) + 1

            calculate_distinct_values_df = pd.DataFrame({col: all_edges[col]})
            distinct_df = self.init_spark().createDataFrame(calculate_distinct_values_df)
            assert distinct_df.distinct().count() == len(all_edges[col])

    def test_get_dual_histogram_bin_edges_single_distinct_value_bucket(self):
        """Test scenario where we have a single bucket"""
        column_dtype_map = {
            'col1': 'int',
            'col2': 'float',
            'col3': 'double',
            'col4': 'decimal',
            'col5': 'string'
        }
        baseline_df = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
            'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
        })
        production_df = pd.DataFrame({
            'col1': [1, 1, 1, 1, 1],
            'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
            'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
        })
        baseline_df = self.init_spark().createDataFrame(baseline_df)
        production_df = self.init_spark().createDataFrame(production_df)
        numerical_columns = get_numerical_cols_with_df(column_dtype_map,
                                                       baseline_df)

        all_edges = get_dual_histogram_bin_edges(
            baseline_df, production_df, baseline_df.count(), production_df.count(), numerical_columns
        )

        assert all_edges is not None
        for col in numerical_columns:
            assert all_edges.get(col, None) is not None

            if col == 'col1':
                assert len(all_edges[col]) == 2
            else:
                assert len(all_edges[col]) == self._num_bins_by_struges_algorithm(baseline_df) + 1

            calculate_distinct_values_df = pd.DataFrame({col: all_edges[col]})
            distinct_df = self.init_spark().createDataFrame(calculate_distinct_values_df)
            assert distinct_df.distinct().count() == len(all_edges[col])

    def init_spark(self):
        """Get or create spark session."""
        spark = SparkSession.builder.appName("test").getOrCreate()
        return spark
