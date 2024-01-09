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


def is_lightGBM_supported_categorical_column(baseline_data, column_name):
    """Determine whether the categorical column is supported by lightGBM.

    :param column_name: the column to determine
    :type column_name: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :rtype: boolean
    """
    baseline_column = pd.Series(baseline_data[column_name])
    baseline_column_type = baseline_column.dtype.name
    # LightGBM cannot accept anything but bool, int and float, only datatime64 can be converted to int
    return pd.api.types.is_datetime64_ns_dtype(baseline_column) or pd.api.types.is_timedelta64_ns_dtype(baseline_column)


def compute_lightgbm_unsupported_categorical_features(baseline_data, target_column, categorical_features):
    """Compute which categorical features are not supported by LightGBM model.

    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :param target_column: the column to predict
    :type target_column: string
    :param categorical_features: The list of categorical features 
    :type categorical_features: list[string]
    :return: lightgbm unsupported categorical features
    :rtype: list[string]
    """
    # LightGBM cannot take categorical features. For data types, it can only take types of bool, int, and float
    lightgbm_unsupported_categorical_features = []
    for column in categorical_features:
        if column != target_column and not is_lightGBM_supported_categorical_column(baseline_data, column):
            lightgbm_unsupported_categorical_features.append(column)
    print("Successfully got feature importance categorical columns")
    return lightgbm_unsupported_categorical_features
