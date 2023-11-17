# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse

import pandas as pd
import numpy as np
from pyspark.sql import DataFrame
from dateutil import parser
import mltable
from mltable import MLTable
import tempfile
import json
from fsspec import AbstractFileSystem
from azureml.fsspec import AzureMachineLearningFileSystem
from datetime import datetime
from pyspark.sql.functions import lit
from shared_utilities.momo_exceptions import DataNotFoundError
from shared_utilities.io_utils import (
    init_spark,
    try_read_mltable_in_spark_with_error,
    save_spark_df_as_mltable,
)
from shared_utilities.event_utils import add_tags_to_root_run
from shared_utilities.constants import (
    MDC_CORRELATION_ID_COLUMN,
    MDC_DATA_COLUMN,
    MDC_DATAREF_COLUMN,
    SCHEMA_INFER_ROW_COUNT,
    AML_MOMO_ERROR_TAG
)

from typing import Tuple
import os
from urllib.parse import urlparse
from azureml.core.run import Run
from azure.ai.ml import MLClient


def _convert_to_azureml_long_form(url_str: str, datastore: str, sub_id=None, rg_name=None, ws_name=None) -> str:
    """Convert path to AzureML path."""
    url = urlparse(url_str)
    if url.scheme in ["https", "http"]:
        idx = url.path.find('/', 1)
        path = url.path[idx+1:]
    elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
        path = url.path[1:]
    elif url.scheme == "azureml" and url.hostname == "datastores":  # azrueml short form
        idx = url.path.find('/paths/')
        path = url.path[idx+7:]
    else:
        return url_str  # azureml long form, azureml asset, file or other scheme, return original path directly

    sub_id = sub_id or os.environ.get("AZUREML_ARM_SUBSCRIPTION", None)
    rg_name = rg_name or os.environ.get("AZUREML_ARM_RESOURCEGROUP", None)
    ws_name = ws_name or os.environ.get("AZUREML_ARM_WORKSPACE_NAME", None)

    return f"azureml://subscriptions/{sub_id}/resourcegroups/{rg_name}/workspaces/{ws_name}/datastores" \
           f"/{datastore}/paths/{path}"


def _get_datastore_from_input_path(input_path: str, ml_client=None) -> str:
    """Get datastore name from input path."""
    url = urlparse(input_path)
    if url.scheme == "azureml":
        if ':' in url.path:  # azureml asset path
            return _get_datastore_from_asset_path(input_path, ml_client)
        else:  # azureml long or short form
            return _get_datastore_from_azureml_path(input_path)
    elif url.scheme == "file" or os.path.isdir(input_path):
        return None  # local path for testing, datastore is not needed
    else:
        raise ValueError("Only azureml path(long, short or asset) is supported as input path of the MDC preprocessor.")


def _get_workspace_info() -> Tuple[str, str, str]:
    """Get workspace info from Run context and environment variables."""
    ws = Run.get_context().experiment.workspace
    sub_id = ws.subscription_id or os.environ.get("AZUREML_ARM_SUBSCRIPTION")
    rg_name = ws.resource_group or os.environ.get("AZUREML_ARM_RESOURCEGROUP")
    ws_name = ws.name or os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    return sub_id, rg_name, ws_name


def _get_datastore_from_azureml_path(azureml_path: str) -> str:
    start_idx = azureml_path.find('/datastores/')
    end_idx = azureml_path.find('/paths/')
    return azureml_path[start_idx+12:end_idx]


def _get_datastore_from_asset_path(asset_path: str, ml_client=None) -> str:
    if not ml_client:
        sub_id, rg_name, ws_name = _get_workspace_info()
        ml_client = MLClient(subscription_id=sub_id, resource_group=rg_name, workspace_name=ws_name)

    # todo: validation
    asset_sections = asset_path.split(':')
    asset_name = asset_sections[1]
    asset_version = asset_sections[2]

    data_asset = ml_client.data.get(asset_name, asset_version)
    return data_asset.datastore or _get_datastore_from_input_path(data_asset.path)


def _raw_mdc_uri_folder_to_mltable(
    start_datetime: datetime, end_datetime: datetime, input_data: str
):
    """Create mltable definition - extract, filter and convert columns."""
    # Extract partition format
    table = mltable.from_json_lines_files(
        paths=[{"pattern": f"{input_data}**/*.jsonl"}]
    )
    # Uppercase HH for hour info
    partitionFormat = "{PartitionDate:yyyy/MM/dd/HH}/{fileName}.jsonl"
    table = table.extract_columns_from_partition_format(partitionFormat)

    # Filter on partitionFormat based on user data window
    filterStr = f"PartitionDate >= datetime({start_datetime.year}, {start_datetime.month}, {start_datetime.day}, " \
                f"{start_datetime.hour}) and PartitionDate <= datetime({end_datetime.year}, {end_datetime.month}, " \
                f"{end_datetime.day}, {end_datetime.hour})"
    table = table.filter(filterStr)

    # Data column is a list of objects, convert it into string because spark.read_json cannot read object
    table = table.convert_column_types({"data": mltable.DataType.to_string()})
    return table


def _convert_mltable_to_spark_df(table: MLTable, preprocessed_input_data: str,
                                 fs: AbstractFileSystem = None, add_tags_func=None) -> DataFrame:
    """
    Convert MLTable to Spark DataFrame.

    A DataNotFoundError will be raised if no data in mltable, otherwise a non empty Spark DataFrame will be returned.
    """
    with tempfile.TemporaryDirectory() as mltable_temp_path:
        # Save MLTable to temp location
        table.save(mltable_temp_path)

        # Save preprocessed_data MLTable to temp location
        des_path = preprocessed_input_data + "temp"
        fs = fs or AzureMachineLearningFileSystem(des_path)  # for testing
        print("MLTable path:", des_path)
        # TODO: Evaluate if we need to overwrite
        # upload the mltable folder to azureml://, so it is available for all executors
        fs.upload(
            lpath=mltable_temp_path,
            rpath=des_path,
            **{"overwrite": "MERGE_WITH_OVERWRITE"},
            recursive=True,
        )

    # Read mltable from preprocessed_data
    try:
        return try_read_mltable_in_spark_with_error(des_path, "preprocessed_data")
    except DataNotFoundError as e:
        tags = {AML_MOMO_ERROR_TAG: "DataNotFoundError"}
        add_tags_func = add_tags_func or add_tags_to_root_run
        add_tags_func(tags)
        raise e


def _get_data_columns(df: DataFrame) -> list:
    columns = []
    if MDC_DATA_COLUMN in df.columns:
        columns.append(MDC_DATA_COLUMN)
    if MDC_DATAREF_COLUMN in df.columns:
        columns.append(MDC_DATAREF_COLUMN)

    return columns


def _extract_data_and_correlation_id(df: DataFrame, extract_correlation_id: bool, datastore: str = None) -> DataFrame:
    """
    Extract data and correlation id from the MDC logs.

    If data column exists, return the json contents in it,
    otherwise, return the dataref content which is a url to the json file.
    """

    def safe_dumps(x):
        if type(x) in [dict, list]:
            return json.dumps(x)
        elif type(x) is np.ndarray:
            return json.dumps(x.tolist())
        else:
            return x

    def convert_object_to_str(dataframe: pd.DataFrame) -> pd.DataFrame:
        columns = dataframe.columns
        for column in columns:
            if dataframe[column].dtype == "object":
                dataframe[column] = dataframe[column].apply(safe_dumps)

        return dataframe

    def read_data(row) -> str:
        data = getattr(row, MDC_DATA_COLUMN, None)
        if data:
            return data

        dataref = getattr(row, MDC_DATAREF_COLUMN, None)
        # convert https path to azureml long form path which can be recognized by azureml filesystem
        # and read by pd.read_json()
        data_url = _convert_to_azureml_long_form(dataref, datastore)
        return data_url
        # TODO: Move this to tracking stream if both data and dataref are NULL

    def row_to_pdf(row) -> pd.DataFrame:
        df = pd.read_json(read_data(row))
        df = convert_object_to_str(df)
        return df

    data_columns = _get_data_columns(df)
    data_rows = df.select(data_columns).rdd.take(SCHEMA_INFER_ROW_COUNT)  # TODO: make it an argument user can define

    spark = init_spark()
    infer_pdf = pd.concat([row_to_pdf(row) for row in data_rows], ignore_index=True)
    data_as_df = spark.createDataFrame(infer_pdf)
    # data_as_df.show()
    # data_as_df.printSchema()

    def extract_data_and_correlation_id(entry, correlationid):
        result = pd.read_json(entry)
        result = convert_object_to_str(result)
        result[MDC_CORRELATION_ID_COLUMN] = ""
        for index, row in result.iterrows():
            result.loc[index, MDC_CORRELATION_ID_COLUMN] = (
                correlationid + "_" + str(index)
            )
        return result

    def transform_df_function_with_correlation_id(iterator):
        for df in iterator:
            yield pd.concat(
                extract_data_and_correlation_id(
                    read_data(row),
                    getattr(row, MDC_CORRELATION_ID_COLUMN),
                )
                for row in df.itertuples()
            )

    def transform_df_function_without_correlation_id(iterator):
        for df in iterator:
            pdf = pd.concat(
                convert_object_to_str(pd.read_json(read_data(row))) for row in df.itertuples()
            )
            yield pdf

    if extract_correlation_id:
        # Add empty column to get the correlationId in the schema
        data_as_df = data_as_df.withColumn(MDC_CORRELATION_ID_COLUMN, lit(""))
        data_columns.append(MDC_CORRELATION_ID_COLUMN)
        transformed_df = df.select(data_columns).mapInPandas(
            transform_df_function_with_correlation_id, schema=data_as_df.schema
        )
    else:
        # TODO: if neither data and dataref move to tracking stream (or throw ModelMonitoringException?)
        transformed_df = df.select(data_columns).mapInPandas(
            transform_df_function_without_correlation_id, schema=data_as_df.schema
        )
    return transformed_df


def _raw_mdc_uri_folder_to_preprocessed_spark_df(
        data_window_start: datetime, data_window_end: datetime,
        input_data: str, preprocessed_input_data: str, extract_correlation_id: bool,
        fs: AbstractFileSystem = None, add_tags_func=None) -> DataFrame:
    """Read raw MDC data, preprocess, and return in a Spark DataFrame."""
    # Parse the dates
    start_datetime = parser.parse(data_window_start)
    end_datetime = parser.parse(data_window_end)

    table = _raw_mdc_uri_folder_to_mltable(start_datetime, end_datetime, input_data)
    # print("MLTable:", table)

    df = _convert_mltable_to_spark_df(table, preprocessed_input_data, fs, add_tags_func)
    # print("df after converting mltable to spark df:")
    # df.select("data").show(truncate=False)
    # df.printSchema()

    datastore = _get_datastore_from_input_path(input_data)
    # print("Datastore:", datastore)
    transformed_df = _extract_data_and_correlation_id(df, extract_correlation_id, datastore)
    # transformed_df.show()
    # transformed_df.printSchema()

    return transformed_df


def mdc_preprocessor(
    data_window_start: str,
    data_window_end: str,
    input_data: str,
    preprocessed_input_data: str,
    extract_correlation_id: bool,
    fs: AbstractFileSystem = None,
):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
        extract_correlation_id: The boolean to extract correlation Id from the MDC logs.
    """
    transformed_df = _raw_mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, input_data,
                                                                  preprocessed_input_data, extract_correlation_id, fs)

    save_spark_df_as_mltable(transformed_df, preprocessed_input_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--extract_correlation_id", type=str)
    parser.add_argument("--preprocessed_input_data", type=str)
    args = parser.parse_args()

    mdc_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_input_data,
        eval(args.extract_correlation_id.capitalize()),
    )


if __name__ == "__main__":
    run()
