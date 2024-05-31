# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

from enum import Enum
import numpy as np
import os
import time
import uuid
import yaml
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential, CredentialUnavailableError
from azure.storage.blob import ContainerClient
from azure.storage.filedatalake import FileSystemClient
from azureml.dataprep.api.errorhandlers import ExecutionError
from azureml.fsspec import AzureMachineLearningFileSystem
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from py4j.protocol import Py4JJavaError
from .constants import MAX_RETRY_COUNT
from shared_utilities.constants import MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE
from shared_utilities.event_utils import post_warning_event
from shared_utilities.momo_exceptions import DataNotFoundError, InvalidInputError
from shared_utilities.store_url import StoreUrl


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
            # TODO: remove this check block after we are able to support submitting managed identity MoMo graphs.
            if isinstance(error, CredentialUnavailableError):
                raise InvalidInputError(MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE.format(message=error.message))
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


def _write_mltable_yaml(mltable_obj, store_url: StoreUrl, credential):
    try:
        content = yaml.dump(mltable_obj, default_flow_style=False)
        print(f"store_url.container_name: {store_url.container_name}")
        print(f"store_url.account_name: {store_url.account_name}")
        print(f"store_url.path: {store_url.path}")
        print(f"store_url.store_type: {store_url.store_type}")
        print(f"store_url._base_url: {store_url._base_url}")
        print(f"store_url._datastore: {store_url._datastore}")
        print(f"store_url.is_local_path(): {store_url.is_local_path()}")

        store_url.write_file(content, "MLTable", True, credential)
        return True
    except Exception as e:
        print(f"Error writing mltable file: {e}")
        return False


def read_mltable_in_spark(mltable_path: str):
    """Read mltable in spark."""
    _verify_mltable_paths(mltable_path)
    spark = init_spark()
    try:
        return spark.read.mltable(mltable_path)
    except ExecutionError as ee:
        if 'AuthenticationError("RuntimeError: Non-matching ' in str(ee):
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, "
                                    "please make sure only data defined in the same AML workspace is used in MLTable.")
    except ValueError as ve:
        if 'AuthenticationError("RuntimeError: Non-matching ' in str(ve):
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, it is not from the same AML workspace.")


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    # We do this first to get Aml OBO credential which will let spark.write.parquet 
    # work in credential-less scenario by initializing certain env variables for free.
    print(f"folder_path: {folder_path}")
    store_url = StoreUrl(folder_path)
    credential = store_url.get_credential()

    metrics_df.write.mode("overwrite").parquet(folder_path)

    base_path = folder_path.rstrip('/')
    output_path_pattern = base_path + "/*.parquet"

    mltable_obj = {
        'paths': [{'pattern': output_path_pattern}],
        'transformations': ['read_parquet']
    }

    retries = 0
    while True:
        if _write_mltable_yaml(mltable_obj, store_url, credential):
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
