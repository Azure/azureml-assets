# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

from enum import Enum
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from shared_utilities.event_utils import post_warning_event
from shared_utilities.momo_exceptions import DataNotFoundError


class NoDataApproach(Enum):
    """Enum for no data approach."""

    IGNORE = 0
    WARNING = 1
    ERROR = 2


def init_spark():
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def _is_not_found_exception(error: Exception):
    return (
        isinstance(error, IndexError)
        or "The requested stream was not found" in error.args[0]
        or "Not able to find MLTable file" in error.args[0]
        or "Partition 0 is out of bounds." in error.args[0]
    )


def try_read_mltable_in_spark_with_warning(mltable_path: str, input_name: str) -> DataFrame:
    """Read mltable in spark. In case of failure, posts a warning to the job and returns None."""
    return try_read_mltable_in_spark(mltable_path, input_name, NoDataApproach.WARNING)


def try_read_mltable_in_spark_with_error(mltable_path: str, input_name: str) -> DataFrame:
    """Read mltable in spark."""
    return try_read_mltable_in_spark(mltable_path, input_name, NoDataApproach.ERROR)


def try_read_mltable_in_spark(mltable_path: str, input_name: str, no_data_approach=NoDataApproach.IGNORE) -> DataFrame:
    """
    Read mltable in spark.

    If data not found, conduct different error handling based on no_data_approach.
    """
    try:
        return read_mltable_in_spark(mltable_path)
    except Exception as error:
        if _is_not_found_exception(error):
            print(error)
            error_message = f"No data was found for input '{input_name}' in the specified window."
            print(error_message)
            if no_data_approach == NoDataApproach.IGNORE:
                return None
            elif no_data_approach == NoDataApproach.WARNING:
                post_warning_event(
                    error_message
                    + " Please visit aka.ms/mlmonitoringhelp for more information."
                )
                return None
            else:  # no_data_approach == NoDataApproach.ERROR:
                raise DataNotFoundError(error_message)
        else:
            raise error


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    spark = init_spark()
    return spark.read.mltable(mltable_path)


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    metrics_df.write.option("output_format", "parquet").option(
        "overwrite", True
    ).mltable(folder_path)


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()
