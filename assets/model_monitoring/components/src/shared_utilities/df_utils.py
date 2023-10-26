# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains additional utilities that are applicable to dataframe."""
import pyspark.sql as pyspark_sql
import pandas as pd


def get_numerical_columns(column_dtype_map: dict, baseline_df) -> list:
    """Get numerical columns from all columns."""
    # NOTE: byte, short, int, long are not included in the list because they are ambiguous with categorical columns.
    # They should be added to the list based on some heuristics or user preference.
    numerical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["float", "double", "decimal"] or column_dtype_map[column] == "int" and is_numerical(pd.Series(baseline_df[column]))
    ]
    return numerical_columns


def get_categorical_columns(column_dtype_map: dict, baseline_df) -> list:
    """Get categorical columns from all columns."""
    # NOTE: byte, short, int, long are not included in the list because they are ambiguous with numerical columns.
    # They should be added to the list based on some heuristics or user preference.
    categorical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["string", "bool"] or column_dtype_map[column] == "int" and is_categorical(pd.Series(baseline_df[column]))
    ]
    return categorical_columns


def get_distinct_ratio(column):
    distinct_values = column.nunique()
    total_values = column.size
    return distinct_values / total_values
    
def is_numerical(column):
    if pd.api.types.is_numeric_dtype(column):
        distinct_value_ratio = get_distinct_ratio(column)
        return distinct_value_ratio >= 0.05
    return False

def is_categorical(column):
    if pd.api.types.is_numeric_dtype(column):
        distinct_value_ratio = get_distinct_ratio(column)
        return distinct_value_ratio < 0.05
    return False
   
        
def get_common_columns(
    baseline_df: pyspark_sql.DataFrame, production_df: pyspark_sql.DataFrame
) -> list:
    """Get common columns from baseline and production dataframes."""
    baseline_df_dtypes = dict(baseline_df.dtypes)
    production_df_dtypes = dict(production_df.dtypes)

    common_columns = {}
    for (column_name, data_type) in baseline_df_dtypes.items():
        if production_df_dtypes.get(column_name) == data_type:
            common_columns[column_name] = data_type
    return common_columns


def select_columns_from_spark_df(df: pyspark_sql.DataFrame, column_list: list):
    """Select comlumns from given spark dataFrame."""
    column_list = list(map(str.strip, column_list))
    df = df.select(column_list)
    return df


def row_has_value(row: pyspark_sql.Row, row_name: str) -> bool:
    """Check if a row has the given column."""
    return row_name in row and row[row_name] is not None and row[row_name] != ""


def add_value_if_present(
    row: pyspark_sql.Row, row_name: str, dict: dict, target_property_name: str
) -> dict:
    """Add value to a dictionary if it is present in a row."""
    if row_has_value(row, row_name):
        dict[target_property_name] = row[row_name]
    return dict
