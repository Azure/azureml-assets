# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the df utilities."""
from shared_utilities.df_utils import (
        get_numerical_cols_with_df,
        get_categorical_cols_with_df,
        is_categorical,
        is_numerical,
        get_common_columns
    )
import pandas as pd
import pytest
from pyspark.sql import SparkSession


@pytest.mark.unit
class TestDFUtils:
    """Test class for df utilities."""

    def test_get_numerical_cols_with_df(self):
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

    def test_get_categorical_columns(self):
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
        column_dtype_map = {
                            'col1': 'int'
                            }
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

    def test_is_categorical(self):
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

    def test_is_numerical(self):
        """Test int is numerical."""
        # Test with integer column with high distinct value ratio
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5])
        result = True
        assert is_numerical(baseline_column) == result

        # Test with integer column with low distinct value ratio
        result = False
        baseline_column = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, None, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert is_numerical(baseline_column) == result

        # Test with integer column with high distinct value ratio
        result = True
        baseline_column = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert is_numerical(baseline_column) == result

    def test_get_common_columns(self):
        # Test with two empty dataframes
        baseline_df = pd.DataFrame(columns=[])
        production_df = pd.DataFrame(columns=[])
        assert get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have no common columns
        baseline_df = pd.DataFrame([(1, "a"), (2, "b")],
                                   columns=["id", "name"])
        production_df = pd.DataFrame([(3, "c"), (4, "d")],
                                     columns=["age",
                                              "gender"])
        assert get_common_columns(baseline_df, production_df) == {}

        # Test with two dataframes that have one common column
        baseline_df = pd.DataFrame([(1, "a"), (2, "b")], columns=["id",
                                                                  "name"])
        production_df = pd.DataFrame([(1, "c"), (2, "d")], columns=["id",
                                                                    "age"])
        assert get_common_columns(baseline_df, production_df) == {"id": "int64"}

        # Test with two dataframes that have multiple common columns
        baseline_df = pd.DataFrame([(1, "a", 10), (2, "b", 20)],
                                   columns=["id", "name", "age"])
        production_df = pd.DataFrame([(1, "c", 30), (2, "d", 40)],
                                     columns=["id", "name", "age"])
        assert get_common_columns(baseline_df, production_df) == {"id": "int64", "name": "object", "age": "int64"}

        # Test with two dataframes that have different Types in common columns
        baseline_df = pd.DataFrame([(1.0, "a", 10), (2, "b", 20)],
                                   columns=["id", "name", "age"])
        production_df = pd.DataFrame([(1, "c", 30), (2, "d", 40)],
                                     columns=["id", "name", "age"])
        assert get_common_columns(baseline_df, production_df) == {"name": "object", "age": "int64"}
        
    def init_spark(self):
        """Get or create spark session."""
        spark = SparkSession.builder.appName("test").getOrCreate()
        return spark
