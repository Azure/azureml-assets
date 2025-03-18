# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to read write data."""

import numpy as np
import time
import traceback
import yaml

from azureml.dataprep.api.errorhandlers import ExecutionError
from azureml.dataprep.api.mltable._mltable_helper import UserErrorException
from enum import Enum
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
from shared_utilities.constants import MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE
from shared_utilities.event_utils import post_warning_event
from shared_utilities.momo_exceptions import DataNotFoundError, InvalidInputError
from shared_utilities.store_url import StoreUrl
from .constants import MAX_RETRY_COUNT


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
            try:
                from azure.ai.ml.identity import CredentialUnavailableError
                if isinstance(error, CredentialUnavailableError):
                    raise InvalidInputError(MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE.format(message=error.message))
            except ModuleNotFoundError:
                print(
                    "Failed to import from module azure-ai-ml to check if we have CredentialUnavailableError. "
                    "Check for LM failure or stale cache being used. Throwing exception as usual.")
            raise error
    return df if df and not df.isEmpty() else process_input_not_found(InputNotFoundCategory.NO_INPUT_IN_WINDOW)


def _write_mltable_yaml(mltable_obj, folder_path: str):
    try:
        store_url = StoreUrl(folder_path)
        content = yaml.dump(mltable_obj, default_flow_style=False)
        store_url.write_file(content, "MLTable", True)
        return True
    except InvalidInputError as iie:
        print(f"Unretriable InvalidInputError writing mltable file: {iie}")
        raise iie
    except Exception:
        print(f"Error writing mltable file: \n{traceback.format_exc()}")
        return False


def read_mltable_in_spark(mltable_path: str) -> DataFrame:
    """Read mltable in spark."""
    if mltable_path is None:
        raise InvalidInputError("MLTable path is None.")
    # validate if we can access the mltable, e.g. if env. is ready to access credential-less data
    store_url = StoreUrl(mltable_path)
    store_url.get_credential(True)  # will raise exception if not able to access

    spark = init_spark()
    try:
        return spark.read.mltable(mltable_path)
    except UserErrorException as ue:
        ue_str = str(ue)
        if 'Not able to find MLTable file' in ue_str:
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, it is not found or not accessible.")
        elif 'MLTable yaml is invalid' in ue_str:
            raise InvalidInputError(f"Invalid MLTable yaml content in {mltable_path}, please make sure all paths "
                                    "defined in MLTable is in correct format and supported scheme.")
        else:
            raise ue
    except ExecutionError as ee:
        ee_str = str(ee)
        if 'AuthenticationError("RuntimeError: Non-matching ' in ee_str:
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, "
                                    "please make sure only data defined in the same AML workspace is used in MLTable.")
        elif 'StreamError(NotFound)' in ee_str and 'The requested stream was not found' in ee_str:
            raise InvalidInputError(f"One or more paths defined in MLTable {mltable_path} is not found.")
        else:
            raise ee
    except ValueError as ve:
        ve_str = str(ve)
        if 'AuthenticationError("RuntimeError: Non-matching ' in ve_str:
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, it is not from the same AML workspace.")
        elif 'StreamError(PermissionDenied(' in ve_str:
            # TODO add link to doc
            raise InvalidInputError(f"No permission to read MLTable {mltable_path}, please make it as credential data."
                                    " Or you can run Model Monitor job with managed identity and grant proper data "
                                    "access permission to the user managed identity attached to this AML workspace.")
        else:
            raise ve
    except RuntimeError as re:
        re_str = str(re)
        if 'Data asset service returned invalid MLTable yaml' in re_str:
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, looks like the MLTable is created with "
                                    "DataSetV1 API, please recreate it with DataSetV2 API. "
                                    "You can do it in the AML studio or with the latest SDK.")
        else:
            raise re
    except SystemError as se:
        if 'Name or service not known' in str(se):
            raise InvalidInputError(f"Failed to read MLTable {mltable_path}, the storage account is not found.")
        else:
            raise se


def save_spark_df_as_mltable(metrics_df, folder_path: str):
    """Save spark dataframe as mltable."""
    base_path = folder_path.rstrip('/')
    output_path_pattern = base_path + "/data/*.parquet"

    mltable_obj = {
        'paths': [{'pattern': output_path_pattern}],
        'transformations': ['read_parquet']
    }

    retries = 0
    while True:
        if _write_mltable_yaml(mltable_obj, folder_path):
            break
        retries += 1
        if retries >= MAX_RETRY_COUNT:
            raise Exception("Failed to write mltable yaml file after multiple retries.")
        time.sleep(1)

    metrics_df.write.mode("overwrite").parquet(base_path+"/data/")


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
