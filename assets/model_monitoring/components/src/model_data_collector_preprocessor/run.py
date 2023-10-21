# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse

import pandas as pd
from pyspark.sql import DataFrame
from dateutil import parser
import mltable
from mltable import MLTable
import tempfile

from model_data_collector_preprocessor.mdc_preprocessor_helper import convert_to_azureml_long_form, get_datastore_from_input_path
from fsspec import AbstractFileSystem
from azureml.fsspec import AzureMachineLearningFileSystem
from datetime import datetime
from pyspark.sql.functions import col, lit
from shared_utilities.event_utils import post_warning_event
from shared_utilities.io_utils import (
    init_spark,
    try_read_mltable_in_spark,
    save_spark_df_as_mltable,
)

from shared_utilities.constants import (
    MDC_CHAT_HISTORY_COLUMN,
    MDC_CORRELATION_ID_COLUMN,
    MDC_DATA_COLUMN,
    MDC_DATAREF_COLUMN
)

def _raw_mdc_uri_folder_to_mltable(
    start_datetime: datetime, end_datetime: datetime, input_data: str
):
    '''Create mltable definition - extract, filter and convert columns.'''
    # Extract partition format
    table = mltable.from_json_lines_files(
        paths=[{"pattern": f"{input_data}**/*.jsonl"}]
    )
    # Uppercase HH for hour info
    partitionFormat = "{PartitionDate:yyyy/MM/dd/HH}/{fileName}.jsonl"
    table = table.extract_columns_from_partition_format(partitionFormat)

    # Filter on partitionFormat based on user data window
    filterStr = f"PartitionDate >= datetime({start_datetime.year}, {start_datetime.month}, {start_datetime.day}, {start_datetime.hour}) and PartitionDate <= datetime({end_datetime.year}, {end_datetime.month}, {end_datetime.day}, {end_datetime.hour})"  # noqa
    table = table.filter(filterStr)

    # Data column is a list of objects, convert it into string because spark.read_json cannot read object
    table = table.convert_column_types({"data": mltable.DataType.to_string()})
    return table

def _convert_mltable_to_spark_df(table: MLTable, preprocessed_input_data: str, fs: AbstractFileSystem) -> DataFrame:
    """Convert MLTable to Spark DataFrame."""
    
    with tempfile.TemporaryDirectory() as mltable_temp_path:
        # Save MLTable to temp location
        table.save(mltable_temp_path)

        # Save preprocessed_data MLTable to temp location
        des_path = preprocessed_input_data + "temp"
        fs = fs or AzureMachineLearningFileSystem(des_path) # for testing
        print("MLTable path:", des_path)
        # TODO: Evaluate if we need to overwrite
        # Richard: why we need to upload the mltable folder to azureml://?
        fs.upload(
            lpath=mltable_temp_path,
            rpath=des_path, #"",
            **{"overwrite": "MERGE_WITH_OVERWRITE"},
            recursive=True,
        )

    # Read mltable from preprocessed_data
    return try_read_mltable_in_spark(des_path, "preprocessed_data")

def _get_data_columns(df: DataFrame) -> list:
    columns = []
    if MDC_DATA_COLUMN in df.columns:
        columns.append(MDC_DATA_COLUMN)
    if MDC_DATAREF_COLUMN in df.columns:
        columns.append(MDC_DATAREF_COLUMN)
    
    return columns

def _extract_data_and_correlation_id(df: DataFrame, extract_correlation_id: bool, datastore: str) -> DataFrame:
    def read_data(row):
        data = getattr(row, MDC_DATA_COLUMN, None)
        if data:
            return data
        
        dataref = getattr(row, MDC_DATAREF_COLUMN, None)
        data_url = convert_to_azureml_long_form(dataref, datastore)
        return data_url
        # TODO: Move this to tracking stream if both data and dataref are NULL

    # Output MLTable
    data_columns = _get_data_columns(df)
    first_data_row = df.select(data_columns).rdd.map(lambda x: x).first()

    spark = init_spark()
    data_as_df = spark.createDataFrame(pd.read_json(read_data(first_data_row)))

    """ The temporary workaround to remove the chat_history column if it exists.
    We are removing the column because the pyspark DF is unable to parse it.
    This version of the MDC is applied only to GSQ.
    """
    if MDC_CHAT_HISTORY_COLUMN in data_as_df.columns:
        data_as_df = data_as_df.drop(col(MDC_CHAT_HISTORY_COLUMN))

    def tranform_df_function_with_correlation_id(iterator):
        for df in iterator:
            yield pd.concat(
                extract_data_and_correlation_id(
                    read_data(row),
                    getattr(row, MDC_CORRELATION_ID_COLUMN),
                )
                for row in df.itertuples()  # noqa
            )

    def extract_data_and_correlation_id(entry, correlationid):
        result = pd.read_json(entry)
        result[MDC_CORRELATION_ID_COLUMN] = ""
        for index, row in result.iterrows():
            result.loc[index, MDC_CORRELATION_ID_COLUMN] = (
                correlationid + "_" + str(index)
            )
        return result

    def transform_df_function_without_correlation_id(iterator):
        for df in iterator:
            yield pd.concat(
                pd.read_json(read_data(row)) for row in df.itertuples()
            )

    data_columns = _get_data_columns(df)
    if extract_correlation_id:
        # Add empty column to get the correlationId in the schema
        data_as_df = data_as_df.withColumn(MDC_CORRELATION_ID_COLUMN, lit(""))
        data_columns.append(MDC_CORRELATION_ID_COLUMN)
        transformed_df = df.select(data_columns).mapInPandas(
            tranform_df_function_with_correlation_id, schema=data_as_df.schema
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
        fs : AbstractFileSystem = None
    ) -> DataFrame:
    '''Read raw MDC data, preprocess, and return in a Spark DataFrame.'''
    # Parse the dates
    start_datetime = parser.parse(data_window_start)
    end_datetime = parser.parse(data_window_end)

    table = _raw_mdc_uri_folder_to_mltable(start_datetime, end_datetime, input_data)

    df = _convert_mltable_to_spark_df(table, preprocessed_input_data, fs)    

    if not df:
        print("Skipping the Model Data Collector preprocessor.")
        post_warning_event(
            "Although data was found, the window for this current run contains no data. "
            + "Please visit aka.ms/mlmonitoringhelp for more information."
        )
        return

    datastore = get_datastore_from_input_path(input_data)
    transformed_df = _extract_data_and_correlation_id(df, extract_correlation_id, datastore)
    
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
    transformed_df = _raw_mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, input_data, preprocessed_input_data, extract_correlation_id, fs)

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
