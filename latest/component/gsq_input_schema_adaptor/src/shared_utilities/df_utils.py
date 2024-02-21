# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains additional utilities that are applicable to dataframe."""
import pyspark.sql as pyspark_sql
from enum import Enum
from shared_utilities.momo_exceptions import InvalidInputError
from shared_utilities.event_utils import post_warning_event


class NoCommonColumnsApproach(Enum):
    """Enum for no common columns approach."""

    IGNORE = 0
    WARNING = 1
    ERROR = 2


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
        column_dtype_map=None) -> list:
    """Get numerical columns from all columns with dataframe."""
    column_dtype_map = dict(df.dtypes) if column_dtype_map is None else column_dtype_map
    feature_type_override_map = get_feature_type_override_map(override_numerical_features,
                                                              override_categorical_features)
    numerical_columns = [
        column
        for column in column_dtype_map
        if is_numerical(column, column_dtype_map, feature_type_override_map, df)
    ]
    return numerical_columns


def get_categorical_cols_with_df_with_override(
        df,
        override_numerical_features,
        override_categorical_features,
        column_dtype_map=None) -> list:
    """Get categorical columns from all columns with dataframe."""
    column_dtype_map = dict(df.dtypes) if column_dtype_map is None else column_dtype_map
    feature_type_override_map = get_feature_type_override_map(override_numerical_features,
                                                              override_categorical_features)
    categorical_columns = [
        column
        for column in column_dtype_map
        if is_categorical(column, column_dtype_map, feature_type_override_map, df)
    ]
    return categorical_columns


def get_numerical_and_categorical_cols(
        df,
        override_numerical_features,
        override_categorical_features,
        column_dtype_map=None):
    """Get numerical and categorical columns from all columns with dataframe."""
    return (get_numerical_cols_with_df_with_override(df,
                                                     override_numerical_features,
                                                     override_categorical_features,
                                                     column_dtype_map),
            get_categorical_cols_with_df_with_override(df,
                                                       override_numerical_features,
                                                       override_categorical_features,
                                                       column_dtype_map))


def get_feature_type_override_map(override_numerical_features: str, override_categorical_features: str) -> dict:
    """Generate feature type override map with key of feature name and value of "numerical"/"categorical"."""
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


def modify_categorical_columns(df: pyspark_sql.DataFrame, categorical_columns: list) -> list:
    """
    Modify categorical columns, filtering out unsupported or non-meaningful columns.

    Args:
    df (pyspark.sql.DataFrame): The input DataFrame
    categorical_columns: List of categorical columns

    Returns:
    modified_categorical_columns: Modified categorical column
    """
    # Only do the data quality check for string type. Ignore all the other types
    # Ignore bool, time, date categorical columns because they are meaningless for data quality calculation
    # Ignore binary because it will throw type not supported error for mode
    modified_categorical_columns = []
    dtype_map = dict(df.dtypes)
    for column in categorical_columns:
        if dtype_map[column] == "string":
            modified_categorical_columns.append(column)
    return modified_categorical_columns


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


def try_get_common_columns_with_warning(
    baseline_df: pyspark_sql.DataFrame, production_df: pyspark_sql.DataFrame
) -> dict:
    """Get common columns. Post warning to the job and return empty dict."""
    return try_get_common_columns(baseline_df, production_df, NoCommonColumnsApproach.WARNING)


def try_get_common_columns_with_error(
    baseline_df: pyspark_sql.DataFrame, production_df: pyspark_sql.DataFrame
) -> dict:
    """Get common columns. Raise error if dictionary is empty."""
    return try_get_common_columns(baseline_df, production_df, NoCommonColumnsApproach.ERROR)


def try_get_common_columns(
    baseline_df: pyspark_sql.DataFrame,
    production_df: pyspark_sql.DataFrame,
    no_common_columns_approach=NoCommonColumnsApproach.IGNORE
) -> dict:
    """
    Compute the common columns between baseline and production dataframes.

    If common columns are not found, conduct different error handling based on no_common_columns_approach.
    """
    common_columns_dict = get_common_columns(baseline_df, production_df)
    if not common_columns_dict:
        error_message = (
            "Found no common columns between input datasets. Try double-checking"
            " if there are common columns between the input datasets."
            " Common columns must have the same names (case-sensitive) and similar data types."
        )
        if no_common_columns_approach == NoCommonColumnsApproach.ERROR:
            raise InvalidInputError(
                error_message
            )
        elif no_common_columns_approach == NoCommonColumnsApproach.WARNING:
            post_warning_event(
                error_message
                + " Please visit aka.ms/mlmonitoringhelp for more information."
            )
            return {}
        # no_common_columns_approach == NoCommonColumnsApproach.IGNORE:
        else:
            return {}
    # returns found common columns.
    return common_columns_dict
