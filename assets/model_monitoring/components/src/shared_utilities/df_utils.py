# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains additional utilities that are applicable to dataframe."""
import pyspark.sql as pyspark_sql

data_type_long_group = ["long", "int", "bigint", "short", "tinyint", "smallint"]
data_type_numerical_group = ["float", "double", "decimal"]
data_type_categorical_group = ["string", "boolean", "timestamp", "date", "binary"]


def is_numerical(column, column_dtype_map: dict, feature_type_override_map: dict, df):
    """Check if int column should be numerical."""
    is_categorical_col = is_categorical(column, column_dtype_map, feature_type_override_map, df)
    return None if is_categorical_col is None else not is_categorical_col


def is_categorical(column, column_dtype_map: dict, feature_type_override_map: dict, df):
    """Check if int column should be categorical."""
    if feature_type_override_map.get(column, None) == "categorical":
        return True
    if feature_type_override_map.get(column, None) == "numerical":
        return False
    if column_dtype_map[column] in data_type_categorical_group:
        return True
    if column_dtype_map[column] in data_type_numerical_group:
        return False
    if column_dtype_map[column] in data_type_long_group:
        distinct_value_ratio = get_distinct_ratio(df.select(column).rdd.flatMap(lambda x: x).collect())
        return distinct_value_ratio < 0.05

    print(f"Unknown column type: {column_dtype_map[column]}, column name: {column}")
    return None


def get_numerical_cols_with_df_with_override(
        df,
        override_numerical_features,
        override_categorical_features,
        column_dtype_map=None,) -> list:
    """Get numerical columns from all columns with dataframe."""
    column_dtype_map = dict(df.dtypes) if column_dtype_map is None else column_dtype_map
    feature_type_override_map = get_feature_type_override_map(override_numerical_features,
                                                              override_categorical_features)
    numerical_columns = [
        column
        for column in column_dtype_map
        if is_numerical(column, column_dtype_map, feature_type_override_map, df) == True
    ]
    return numerical_columns


def get_categorical_cols_with_df_with_override(
        df,
        override_numerical_features,
        override_categorical_features,
        column_dtype_map=None,) -> list:
    """Get categorical columns from all columns with dataframe."""
    column_dtype_map = dict(df.dtypes) if column_dtype_map is None else column_dtype_map
    feature_type_override_map = get_feature_type_override_map(override_numerical_features,
                                                              override_categorical_features)
    categorical_columns = [
        column
        for column in column_dtype_map
        if is_categorical(column, column_dtype_map, feature_type_override_map, df) == True
    ]
    return categorical_columns


def get_numerical_and_categorical_cols(
        df,
        override_numerical_features,
        override_categorical_features,
        column_dtype_map=None,):
    return (get_numerical_cols_with_df_with_override(df,
                                                     column_dtype_map,
                                                     override_numerical_features,
                                                     override_categorical_features),
            get_categorical_cols_with_df_with_override(df,
                                                       column_dtype_map,
                                                       override_numerical_features,
                                                       override_categorical_features))


def get_feature_type_override_map(override_numerical_features: str, override_categorical_features: str) -> dict:
    """ Generate feature type override map with key of feature name and value of "numerical"/"categorical"."""
    feature_type_override_map = {}
    if override_categorical_features:
        for cat_feature in override_categorical_features.split(','):
            feature_type_override_map[cat_feature] = "categorical"
    if override_numerical_features:
        for num_feature in override_numerical_features.split(','):
            feature_type_override_map[num_feature] = "numerical"
    return feature_type_override_map


def get_distinct_ratio(column):
    """Get distict ratio for values in a column."""
    distinct_values = len(set(column))
    total_values = len(column)
    return distinct_values / total_values


def get_common_columns(
    baseline_df: pyspark_sql.DataFrame, production_df: pyspark_sql.DataFrame
) -> dict:
    """Get common columns from baseline and production dataframes."""
    baseline_df_dtypes = dict(baseline_df.dtypes)
    production_df_dtypes = dict(production_df.dtypes)
    common_columns = {}
    for (column_name, data_type) in baseline_df_dtypes.items():
        if production_df_dtypes.get(column_name) == data_type:
            common_columns[column_name] = data_type
        else:
            # if baseline and target are of different type
            # and both of them are in [double, float],
            # We consider them to be double
            if production_df_dtypes.get(column_name) in data_type_numerical_group \
                    and baseline_df_dtypes.get(column_name) in data_type_numerical_group:
                common_columns[column_name] = 'double'
            # if baseline and target are of different type
            # and both of them are in [int, long, short]
            # We consider them to be long
            elif production_df_dtypes.get(column_name) in data_type_long_group\
                    and baseline_df_dtypes.get(column_name) in data_type_long_group:
                common_columns[column_name] = 'long'

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
