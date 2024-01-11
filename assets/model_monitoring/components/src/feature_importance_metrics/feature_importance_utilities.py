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


def is_lgbm_supported_categorical_column(baseline_data, column_name):
    """Determine whether the categorical column is supported by lightGBM.

    :param column_name: the column to determine
    :type column_name: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :rtype: boolean
    """
    baseline_column = pd.Series(baseline_data[column_name])
    # datetime and timedelta (except datetime_ns and timedelta_ns) can be converted to an integer (this is what lightgbm does internally) 
    if pd.api.types.is_datetime64_ns_dtype(baseline_column) or pd.api.types.is_timedelta64_ns_dtype(baseline_column):
        return False
    if pd.api.types.is_datetime64_dtype(baseline_column) or pd.api.types.is_timedelta64_dtype(baseline_column):
        return True
    return False


def compute_categorical_features_lgbm(baseline_data, target_column, categorical_features):
    """Modify the common categorical features to get a final categorical list for LightGBM model to ignore

    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :param target_column: the column to predict
    :type target_column: string
    :param categorical_features: The list of categorical features 
    :type categorical_features: list[string]
    :return: lightgbm categorical features
    :rtype: list[string]
    """
    # Lightgbm can only support features that can be converted to bool, int, float.
    # If these features can't be converted, we have to mark them as "category" types so lightgbm will ignore them.
    # In our design, we mark all known categorical features (including bool) as "category", 
    # only filter out datetime and timedelta because they can be converted to int
    categorical_features_lgbm = []
    for column in categorical_features:
        if column != target_column and not is_lgbm_supported_categorical_column(baseline_data, column):
            categorical_features_lgbm.append(column)
    print("Successfully got feature importance categorical columns")
    return categorical_features_lgbm
