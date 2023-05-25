# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for feature attribution drift component."""
import pandas as pd
import logging
from shared_utilities.io_utils import init_spark

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def convert_pandas_to_spark(pandas_data):
    """Convert pandas.Dataframe to pySpark.Dataframe.

    :param pandas_data: the input pandas data to convert
    :type pandas_data: pandas.Dataframe
    :return: the input data in spark format
    :rtype: pySpark.Dataframe
    """
    spark = init_spark()
    return spark.createDataFrame(pandas_data)


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
        baseline_column = pd.Series(baseline_data[column])
        if baseline_column.name != target_column:
            column_type = baseline_column.dtype.name
            if column_type == "object" or column_type == "bool":
                categorical_features.append(baseline_column.name)
            # if the type is int and the ratio of distinct values to total values
            # is less than .05 than the column is considered categorical
            elif column_type == "int64":
                distinct_column_values = len(baseline_column.unique())
                total_column_values = len(baseline_column)
                distinct_value_ratio = distinct_column_values / total_column_values
                if distinct_value_ratio < 0.05:
                    categorical_features.append(baseline_column.name)
    _logger.info("Successfully categorized columns")
    return categorical_features
