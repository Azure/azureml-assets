"""This file contains the core logic for feature attribution drift component."""
import pandas as pd
from datetime import datetime

from azureml.metrics import constants
from pyspark.sql.types import (
    DoubleType,
    StructType,
    StructField,
    StringType,
)
from shared_utilities.io_utils import init_spark, save_spark_df_as_mltable


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


def write_to_mltable(metrics_artifacts, output_data_file_name):
    """

    Args:
        metrics_artifacts:
        output_data_file_name:

    Returns:

    """
    log_time_and_message("Begin writing metric to mltable")
    # ToDo: We might need to support this for vector metrices as well
    metrics_data = pd.DataFrame([metrics_artifacts[constants.Metric.Metrics]])
    spark_data = convert_pandas_to_spark(metrics_data)
    save_spark_df_as_mltable(spark_data, output_data_file_name)


def construct_signal_metrics(
        metrics_artifacts,
        output_data_file_name,
        regression_rmse_threshold,
        regression_meanabserror_threshold,
        classification_precision_threshold,
        classification_accuracy_threshold,
        classification_recall_threshold
        ):
    """
    Args:
        metrics_artifacts:
        output_data_file_name:

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
        "accuracy": "Accuracy",
        "precision_score_macro": "Precision",
        "recall_score_macro": "Recall",
        "mean_absolute_error": "MeanAbsoluteError",
        "root_mean_squared_error": "RootMeanSquaredError"       
    }
    metrics_data_pd = pd.DataFrame([metrics_artifacts[constants.Metric.Metrics]]).reset_index()
    metrics_data_pd.iteritems = metrics_data_pd.items
    metrics_data_df = convert_pandas_to_spark(metrics_data_pd)
    # remove the index column
    metrics_data_df = metrics_data_df.drop("index")
    metrics_data_df.show()
    # construct the signal output schema
    schema = StructType([
            StructField("metric_name", StringType(), True),
            StructField("metric_value", DoubleType(), True),
            StructField("threshold_value", StringType(), True),
    ])
    spark = init_spark()
    metric_names = metrics_data_df.columns

    signal_output_df = spark.createDataFrame(
                [(metrics_name_to_output_metrics_name_map[col_],
                  metrics_data_df.first()[col_],
                  metrics_name_to_threshold_map[col_]) for col_ in metrics_data_df.columns],
                 schema)

    save_spark_df_as_mltable(signal_output_df, output_data_file_name)
