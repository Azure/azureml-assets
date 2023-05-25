# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data in pyspark."""

import pyspark.sql as pyspark_sql
from pyspark.sql.types import StructType, StructField, StringType, FloatType
from shared_utilities.io_utils import (
    init_spark,
    read_mltable_in_spark,
    save_spark_df_as_mltable,
)


def _get_output_spark_df_schema():
    """Get Output DataFrame Schema."""
    schema = StructType(
        [
            StructField("feature_name", StringType(), True),
            StructField("metric_value", FloatType(), True),
            StructField("data_type", StringType(), True),
            StructField("metric_name", StringType(), True),
            StructField("threshold_value", FloatType(), True),
        ]
    )
    return schema


def get_output_spark_df(rows: list):
    """Create Output Spark DataFrame."""
    output_schema = _get_output_spark_df_schema()
    spark = init_spark()
    return spark.createDataFrame(data=rows, schema=output_schema)


def output_computed_measures_tests(metrics_df: pyspark_sql.DataFrame, folder_path: str):
    """Write Data Drift metrics to storage."""
    # TODO: log metrics to mlflow
    save_spark_df_as_mltable(metrics_df, folder_path)


def select_columns_from_spark_df(mltable_path: str, column_list: list):
    """Select comlumns from given spark dataFrame."""
    column_list = list(map(str.strip, column_list))
    df = read_mltable_in_spark(mltable_path)
    df = df.select(column_list)
    return df
