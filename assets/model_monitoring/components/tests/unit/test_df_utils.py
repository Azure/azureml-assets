# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from shared_utilities.df_utils import (
    get_numerical_columns,
    get_categorical_columns,
    is_categorical,
    is_numerical
)
import pandas as pd


def test_get_numerical_columns():
    """Test numerical columns"""
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
    numerical_columns = get_numerical_columns(column_dtype_map, baseline_df)
    assert numerical_columns == ['col1', 'col2', 'col3', 'col4']


def test_get_categorical_columns():
    """Test categorical columns"""
    # Test with all numerical columns
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
    categorical_columns = get_categorical_columns(column_dtype_map,
                                                  baseline_df)
    assert categorical_columns == []

    # Test with int being same value
    column_dtype_map = {
                        'col1': 'int'
                        }
    baseline_df = pd.DataFrame({
                                'col1': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                                })
    categorical_columns = get_categorical_columns(column_dtype_map,
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
    categorical_columns = get_categorical_columns(column_dtype_map,
                                                  baseline_df)
    assert categorical_columns == ['col1', 'col2']


def test_is_categorical():
    """Test int is categorical"""
    # Test with integer column
    result = False
    baseline_column = pd.Series([1, 2, 3, 4, 5])
    assert is_categorical(baseline_column) == result

    # Test with integer column with low distinct value ratio
    baseline_column = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert is_categorical(baseline_column) == result

    # Test with integer column with high distinct value ratio
    baseline_column = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert is_categorical(baseline_column) == result


def test_is_numerical():
    """Test int is numerical"""
    # Test with integer column with high distinct value ratio
    baseline_column = pd.Series([1, 2, 3, 4, 5])
    result = True
    assert is_numerical(baseline_column) == result

    # Test with integer column with low distinct value ratio
    baseline_column = pd.Series([1, 1, 1, 1, 1])
    assert is_numerical(baseline_column) == result
