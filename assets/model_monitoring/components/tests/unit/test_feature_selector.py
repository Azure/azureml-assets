# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the df utilities."""

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    StructField,
    StructType)
from pyspark.sql import SparkSession
from shared_utilities.df_utils import (
        get_numerical_cols_with_df,
        get_categorical_cols_with_df,
        is_categorical,
        is_numerical,
        get_common_columns
    )
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pandas as pd
import pytest

input_1 = []
input_2 = []

@pytest.mark.unit
class TestFeatureSelector:
    """Test class for feature selector."""

    def test_feature_selector_all(self):
        """Test numerical columns."""
        # Test with mixed columns
        column_dtype_map = {'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal',
                            'col5': 'string'
                            }
        baseline_df = pd.DataFrame({
                                    'col1': [1, 2, 2, 4, 4],
                                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                                    'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                                    'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
                                    })
        baseline_df = self.init_spark().createDataFrame(baseline_df)
        numerical_columns = get_numerical_cols_with_df(column_dtype_map,
                                                       baseline_df)
        assert numerical_columns == ['col1', 'col2', 'col3', 'col4']

    def test_selector_subset(self):
        """Test categorical columns."""
        # Test with all numerical columns
        spark = self.init_spark()
        column_dtype_map = {
                            'col1': 'int',
                            'col2': 'float',
                            'col3': 'double',
                            'col4': 'decimal'
                            }
        baseline_df = pd.DataFrame({
                                    'col1': [1, 2, 2, 4, 4],
                                    'col2': [1.1, 2.2, 3.3, 4.4, 5.5],
                                    'col3': [1.11, 2.22, 3.33, 4.44, 5.55],
                                    'col4': [1.111, 2.222, 3.333, 4.444, 5.555]
                                    })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == []

        # Test with int being same value
        column_dtype_map = {'col1': 'int'}
        baseline_df = pd.DataFrame({
                                    'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                             1]
                                    })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == ['col1']

        # Test with all categorical columns
        column_dtype_map = {
            'col1': 'string',
            'col2': 'bool'
        }
        baseline_df = pd.DataFrame({
            'col1': ['a', 'b', 'c', 'd', 'e'],
            'col2': [True, False, True, False, True]
        })
        baseline_df = spark.createDataFrame(baseline_df)
        categorical_columns = get_categorical_cols_with_df(column_dtype_map,
                                                           baseline_df)
        assert categorical_columns == ['col1', 'col2']

    def test_selector_top_by_attribution(self):
        """Test int is categorical."""
        # Test with integer column
        result = False
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5])
        assert is_categorical(baseline_column) == result

        # Test with integer column with low distinct value ratio
        result = True
        baseline_column = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert is_categorical(baseline_column) == result

        # Test with integer column with high distinct value ratio
        result = False
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert is_categorical(baseline_column) == result
