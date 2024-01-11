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


def mark_categorical_column(baseline_df, target_column, categorical_features_lgbm, numerical_features):
    """Mark the categorical column (except target column) type as "category" so lightgbm will ignore them

    :param baseline_df: The baseline data meaning the data used to create the
    model monitor
    :type baseline_df: pandas.DataFrame
    :param target_column: the target column name
    :type target_column: string
    """
    for column in baseline_df.columns:
        col = pd.Series(baseline_df[column])
        if column in categorical_features_lgbm:
            baseline_df[column] = baseline_df[column].astype('category')
        if column not in categorical_features_lgbm and column not in numerical_features and column != target_column:
            log_time_and_message(f"Unknown column: {column}, defult to category.")
            baseline_df[column] = baseline_df[column].astype('category')

