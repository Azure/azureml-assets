# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from shared_utilities.event_utils import post_warning_event


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def try_read_mltable_in_spark_with_warning(mltable_path: str, input_name: str) -> DataFrame:
    """Read mltable in spark."""
    try:
        return read_mltable_in_spark(mltable_path)
    except:
        error_message = f"No data was found for input '{input_name}'."
        print(error_message)
        post_warning_event(error_message
            + " Please visit aka.ms/mlmonitoringhelp for more information."
        )
        return None
    
def try_read_mltable_in_spark(mltable_path: str, input_name: str) -> DataFrame:
    """Read mltable in spark."""
    try:
        return read_mltable_in_spark(mltable_path)
    except:

        print(f"No data was found for input '{input_name}'.")
        return None


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    try:
        return spark.read.mltable(mltable_path)
    except:
        raise Exception(f"Input '{mltable_path}' contains not data. Please visit aka.ms/mlmonitoringhelp for more information.")

def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(folder_path)


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()
