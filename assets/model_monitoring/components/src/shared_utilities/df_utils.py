# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains additional utilities that are applicable to dataframe."""
import pyspark.sql as pyspark_sql


def get_numerical_columns(column_dtype_map: dict) -> list:
    """Get numerical columns from all columns."""
    # NOTE: byte, short, long are not included in the list because they
    # are ambiguous with categorical columns. They should be added to the list
    # based on some heuristics or user preference.
    numerical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["float", "double", "decimal"]
    ]
    return numerical_columns


def get_categorical_columns(column_dtype_map: dict) -> list:
    """Get categorical columns from all columns."""
    # NOTE: byte, short, long are not included in the list because they
    # are ambiguous with numerical columns. They should be added to the
    # list based on some heuristics or user preference.
    categorical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["string", "bool"]
    ]
    return categorical_columns


def get_numerical_cols_with_df(column_dtype_map: dict, baseline_df) -> list:
    """Get numerical columns from all columns with dataframe."""
    # NOTE: byte, short, long are not included in the list because they
    # are ambiguous with categorical columns. They should be added to the list
    # based on some heuristics or user preference.
    numerical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["float", "double", "decimal"]
        or (column_dtype_map[column] in ["int", "bigint", "short", "long"]
            and is_numerical(baseline_df.select(column)
                             .rdd.flatMap(lambda x: x).collect()))
    ]
    return numerical_columns


def get_categorical_cols_with_df(column_dtype_map: dict, baseline_df) -> list:
    """Get categorical columns from all columns with dataframe."""
    # NOTE: byte, short, long are not included in the list because they
    # are ambiguous with numerical columns. They should be added to the
    # list based on some heuristics or user preference.
    categorical_columns = [
        column
        for column in column_dtype_map
        if column_dtype_map[column] in ["string", "bool"]
        or (column_dtype_map[column] in ["int", "bigint", "short", "long"]
            and is_categorical(baseline_df.select(column)
                               .rdd.flatMap(lambda x: x).collect()))
    ]
    return categorical_columns


def get_distinct_ratio(column):
    """Get distict ratio for values in a column."""
    distinct_values = len(set(column))
    total_values = len(column)
    return distinct_values / total_values


def is_numerical(column):
    """Check if int column should be numerical."""
    distinct_value_ratio = get_distinct_ratio(column)
    return distinct_value_ratio >= 0.05


def is_categorical(column):
    """Check if int column should be categorical."""
    distinct_value_ratio = get_distinct_ratio(column)
    return distinct_value_ratio < 0.05


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
