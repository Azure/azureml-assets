# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utility functions for model performance metrics."""

import pandas as pd
from datetime import datetime

from azureml.metrics import constants
from pyspark.sql.types import (
    DoubleType,
    StructType,
    StructField,
    StringType,
)
from shared_utilities.constants import (
    ACCURACY_METRIC_NAME,
    PERCISION_METRIC_NAME,
    RECALL_METRIC_NAME,
    MEAN_ABSOLUTE_ERROR_METRIC_NAME,
    ROOT_MEAN_SQUARED_ERROR_METRIC_NAME,
    SIGNAL_METRICS_GROUP,
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
)
from shared_utilities.io_utils import init_spark, save_spark_df_as_mltable


def log_time_and_message(message):
    """Print the time in addition to message for logging purposes.

    : param message: The message to be printed after the time
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


def construct_signal_metrics(
        metrics_artifacts,
        output_data_file_name,
        predictions_column_name,
        regression_rmse_threshold=None,
        regression_meanabserror_threshold=None,
        classification_precision_threshold=None,
        classification_accuracy_threshold=None,
        classification_recall_threshold=None,
        ):
    """
        Construct the signal metrics.

    Args:
        metrics_artifacts: metrics artifacts
        output_data_file_name: output data file name
        predictions_column_name: prediction column name
        regression_rmse_threshold: rmse threshold
        regression_meanabserror_threshold: mean abs error threshold
        classification_precision_threshold: precision threshold
        classification_accuracy_threshold: accuracy threshold
        classification_recall_threshold: recall threshold
    Returns:
    """
    metrics_name_to_threshold_map = {
        "accuracy": classification_accuracy_threshold,
        "precision_score_macro": classification_precision_threshold,
        "recall_score_macro": classification_recall_threshold,
        "mean_absolute_error": regression_meanabserror_threshold,
        "root_mean_squared_error": regression_rmse_threshold
    }
    metrics_name_to_output_metrics_name_map = {
        "accuracy": ACCURACY_METRIC_NAME,
        "precision_score_macro": PERCISION_METRIC_NAME,
        "recall_score_macro": RECALL_METRIC_NAME,
        "mean_absolute_error": MEAN_ABSOLUTE_ERROR_METRIC_NAME,
        "root_mean_squared_error": ROOT_MEAN_SQUARED_ERROR_METRIC_NAME
    }
    metrics_data_pd = pd.DataFrame([metrics_artifacts[constants.Metric.Metrics]]).reset_index()
    # spark 3.3 cannot convert pandas to spark by "createDataFrame" because no iteritems in panda dataframe
    # The workaround here is to assign iteritems manuals with items
    # when we update to spark 3.4.1, we do not need this workaround anymore
    metrics_data_pd.iteritems = metrics_data_pd.items
    metrics_data_df = convert_pandas_to_spark(metrics_data_pd)
    # remove the index column
    metrics_data_df = metrics_data_df.drop("index")
    metrics_data_df.show()
    # construct the signal output schema
    schema = StructType([
            StructField(SIGNAL_METRICS_METRIC_NAME, StringType(), True),
            StructField(SIGNAL_METRICS_METRIC_VALUE, DoubleType(), True),
            StructField(SIGNAL_METRICS_THRESHOLD_VALUE, StringType(), True)
    ])
    spark = init_spark()

    signal_output_df = spark.createDataFrame(
                [(metrics_name_to_output_metrics_name_map[col_],
                  metrics_data_df.first()[col_],
                  metrics_name_to_threshold_map[col_]) for col_ in metrics_data_df.columns],
                schema)

    save_spark_df_as_mltable(signal_output_df, output_data_file_name)
