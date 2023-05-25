# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import numpy as np
from pyspark.sql import SparkSession


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    df = spark.read.mltable(mltable_path)
    return df


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(folder_path)


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()
