# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for feature attribution drift component."""
import pandas as pd
from datetime import datetime
from shared_utilities.io_utils import init_spark


def log_time_and_message(message):
    """Print the time in addition to message for logging purposes.

    :param message: The message to be printed after the time
    : type message: string
    """
    print(f"[{datetime.now()}] {message}")


def convert_pandas_to_spark(pandas_data):
    """Convert pandas.Dataframe to pySpark.Dataframe.

    :param pandas_data: the input pandas data to convert
    :type pandas_data: pandas.Dataframe
    :return: the input data in spark format
    :rtype: pySpark.Dataframe
    """
    spark = init_spark()
    return spark.createDataFrame(pandas_data)


def is_categorical_column(baseline_data, column_name):
    """Determine whether the column is categorical.

    :param column_name: the column to determine
    :type column_name: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :rtype: boolean
    """
    baseline_column = pd.Series(baseline_data[column_name])
    baseline_column_type = baseline_column.dtype.name
    if (pd.api.types.is_float_dtype(baseline_column) or
            pd.api.types.is_datetime64_ns_dtype(baseline_column) or
            pd.api.types.is_timedelta64_ns_dtype(baseline_column)):
        return False
    # treat all datetime types as categorical since LightGBM cannot accept anything but bool, int and float
    if (pd.api.types.is_object_dtype(baseline_column) or pd.api.types.is_string_dtype(baseline_column)
            or baseline_column_type == "bool" or pd.api.types.is_datetime64_dtype(baseline_column) or
            pd.api.types.is_timedelta64_dtype(baseline_column)):
        return True
    if pd.api.types.is_integer_dtype(baseline_column):
        # if there are more unique values, not likely to be categorical
        distinct_column_values = len(baseline_column.unique())
        total_column_values = len(baseline_column)
        distinct_value_ratio = distinct_column_values / total_column_values
        if distinct_value_ratio < 0.05:
            return True
        else:
            return False
    # Log the datatype detected and default to true
    log_time_and_message(f"Unknown column type: {baseline_column_type}")
    return True


def compute_categorical_features(baseline_data, target_column):
    """Compute which features are categorical based on data type of the columns.

    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :param target_column: the column to predict
    :type target_column: string
    :return: categorical features
    :rtype: list[string]
    """
    categorical_features = []
    for column in baseline_data.columns:
        if column != target_column:
            if is_categorical_column(baseline_data, column):
                categorical_features.append(column)
    print("Successfully categorized columns")
    return categorical_features
