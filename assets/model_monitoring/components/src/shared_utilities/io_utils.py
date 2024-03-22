# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

from enum import Enum
import time
import yaml
import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from .constants import MAX_RETRY_COUNT
from shared_utilities.event_utils import post_warning_event
from shared_utilities.momo_exceptions import DataNotFoundError, InvalidInputError
from model_data_collector_preprocessor.store_url import StoreUrl  # TODO: move StoreUrl to share_utilities


class NoDataApproach(Enum):
    """Enum for no data approach."""

    IGNORE = 0
    WARNING = 1
    ERROR = 2


class InputNotFoundCategory(Enum):
    """Enum for input not found category."""

    NOT_INPUT_MISSING = 0
    NO_INPUT_IN_WINDOW = 1
    ROOT_FOLDER_NOT_FOUND = 2
    MLTABLE_NOT_FOUND = 3
    GENERAL = 10


def init_spark() -> SparkSession:
    """Get or create spark session."""
    spark = SparkSession.builder.appName("AccessParquetFiles").getOrCreate()
    return spark


def _get_input_not_found_category(error: Exception):
    err_msg = error.args[0] if len(error.args) > 0 else ""
    if not isinstance(err_msg, str):
        err_msg = ""
    if isinstance(error, IndexError) or ("Partition 0 is out of bounds." in err_msg):
        return InputNotFoundCategory.NO_INPUT_IN_WINDOW
    elif "The requested stream was not found" in err_msg:
        return InputNotFoundCategory.ROOT_FOLDER_NOT_FOUND
    elif "Not able to find MLTable file" in err_msg:
        return InputNotFoundCategory.MLTABLE_NOT_FOUND
    else:
        return InputNotFoundCategory.NOT_INPUT_MISSING


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
    def process_input_not_found(input_not_found_category: InputNotFoundCategory):
        err_msg_map = {
            InputNotFoundCategory.NOT_INPUT_MISSING:
                "Not input missing error.",
            InputNotFoundCategory.NO_INPUT_IN_WINDOW: (
                f"No data is found for input '{input_name}' in the specified window, "
                "most likely there is no request in the specified window. "
                f"You can check the filter field of MLTable in {mltable_path} for the time window."
            ),
            InputNotFoundCategory.ROOT_FOLDER_NOT_FOUND: (
                f"No data is found for input '{input_name}', "
                "seems the root folder of the input is moved or deleted. "
                f"You can check the pattern field of the MLTable in {mltable_path} for the root folder."
            ),
            InputNotFoundCategory.MLTABLE_NOT_FOUND: (
                "No data is found for input '{input_name}', "
                f"There is no MLTable file in the specified folder {mltable_path}, or even the folder is not exists."
            ),
            InputNotFoundCategory.GENERAL:
                f"No data is found for input '{input_name}'."
        }
        error_message = err_msg_map.get(input_not_found_category, "read mltable failed.")
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

    try:
        df = read_mltable_in_spark(mltable_path)
    except Exception as error:
        input_not_found_category = _get_input_not_found_category(error)
        if input_not_found_category != InputNotFoundCategory.NOT_INPUT_MISSING:
            return process_input_not_found(input_not_found_category)
        else:
            raise error
    return df if df and not df.isEmpty() else process_input_not_found(InputNotFoundCategory.NO_INPUT_IN_WINDOW)


def _verify_mltable_paths(mltable_path: str, ws=None, mltable_dict: dict = None):
    """Verify paths in mltable is supported."""
    mltable_dict = mltable_dict or yaml.safe_load(StoreUrl(mltable_path, ws).read_file_content("MLTable"))
    for path in mltable_dict.get("paths", []):
        path_val = path.get("file") or path.get("folder") or path.get("pattern")
        try:
            path_url = StoreUrl(path_val, ws)  # path_url itself must be valid
            if not path_url.is_local_path():   # and it must be either local(absolute or relative) path
                _ = path_url.get_credential()  # or credential azureml path
        except InvalidInputError as iie:
            raise InvalidInputError(f"Invalid or unsupported path {path_val} in MLTable {mltable_path}") from iie


def _write_mltable_yaml(mltable_obj, dest_path, file_system = None):
    try:
        import os
        import uuid
        from azureml.fsspec import AzureMachineLearningFileSystem

        folder_name = str(uuid.uuid4())
        folder_path = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder_path)
        source_path = os.path.join(folder_path, "MLTable")
        with open(source_path, "w") as yaml_file:
            yaml.dump(mltable_obj, yaml_file)

        fs = file_system or AzureMachineLearningFileSystem(dest_path)
        fs.upload(
            lpath=source_path,
            rpath=dest_path,
            **{"overwrite": "MERGE_WITH_OVERWRITE"},
            recursive=True,
        )
        return True
    except Exception as e:
        print(f"Error writing mltable file: {e}")
        return False


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    _verify_mltable_paths(mltable_path)
    spark = init_spark()
    return spark.read.mltable(mltable_path)


def save_spark_df_as_mltable(metrics_df, folder_path: str, file_system = None):
    """Save spark dataframe as mltable."""
    metrics_df.write.mode("overwrite").parquet(folder_path)

    base_path = folder_path.rstrip('/')
    output_path_pattern = base_path + "/*.parquet"

    mltable_obj = {
        'paths': [{'pattern': output_path_pattern}],
        'transformations': ['read_parquet']
    }

    retries = 0
    while True:
        if _write_mltable_yaml(mltable_obj, folder_path, file_system):
            break
        retries += 1
        if retries >= MAX_RETRY_COUNT:
            raise Exception("Failed to write mltable yaml file after multiple retries.")
        time.sleep(1)


def np_encoder(object):
    """Json encoder for numpy types."""
    if isinstance(object, np.generic):
        return object.item()


def create_spark_df(rows: list, schema: StructType):
    """Create Spark DataFrame."""
    spark = init_spark()
    return spark.createDataFrame(data=rows, schema=schema)


def save_empty_dataframe(schema: StructType, output_path: str):
    """Save empty Data Spark DataFrame."""
    df = create_spark_df([], schema)
    save_spark_df_as_mltable(df, output_path)
