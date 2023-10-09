# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse
import pandas as pd
import mltable
import tempfile
from azureml.fsspec import AzureMachineLearningFileSystem
from datetime import datetime
from pyspark.sql.functions import col, lit
from shared_utilities.datetime_utils import parse_datetime_from_string
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
)


def _convert_uri_folder_to_mltable(
    start_datetime: datetime, end_datetime: datetime, input_data: str
):
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


def mdc_preprocessor(
    data_window_start: str,
    data_window_end: str,
    input_data: str,
    preprocessed_input_data: str,
    extract_correlation_id: bool,
):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        production_data: The data asset on which the date filter is applied.
        monitor_current_time: The current time of the window (inclusive).
        window_size_in_days: Number of days from current time to start time window (exclusive).
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
        extract_correlation_id: The boolean to extract correlation Id from the MDC logs.
    """
    # Format the dates
    format_data = "%Y-%m-%d %H:%M:%S"
    start_datetime = parse_datetime_from_string(format_data, data_window_start)
    end_datetime = parse_datetime_from_string(format_data, data_window_end)

    # Create mltable definition - extract, filter and convert columns.
    table = _convert_uri_folder_to_mltable(start_datetime, end_datetime, input_data)

    # Create MLTable in different location
    save_path = tempfile.mktemp()
    table.save(save_path)

    # Save preprocessed_data MLTable to temp location
    des_path = preprocessed_input_data + "temp"
    fs = AzureMachineLearningFileSystem(des_path)
    print("MLTable path:", des_path)
    # TODO: Evaluate if we need to overwrite
    fs.upload(
        lpath=save_path,
        rpath="",
        **{"overwrite": "MERGE_WITH_OVERWRITE"},
        recursive=True,
    )

    # Read mltable from preprocessed_data
    df = try_read_mltable_in_spark(des_path, "preprocessed_data")

    if not df:
        print("Skipping the Model Data Collector preprocessor.")
        post_warning_event(
            "Although data was found, the window for this current run contains no data. "
            + "Please visit aka.ms/mlmonitoringhelp for more information."
        )
        return

    # Output MLTable
    first_data_row = df.select(MDC_DATA_COLUMN).rdd.map(lambda x: x).first()

    spark = init_spark()
    data_as_df = spark.createDataFrame(pd.read_json(first_data_row[MDC_DATA_COLUMN]))

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
                    getattr(row, MDC_DATA_COLUMN),
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
                pd.read_json(getattr(row, MDC_DATA_COLUMN)) for row in df.itertuples()
            )

    if extract_correlation_id:
        # Add emtpy column to get the correlationId in the schema
        data_as_df = data_as_df.withColumn(MDC_CORRELATION_ID_COLUMN, lit(""))
        transformed_df = df.select(
            MDC_DATA_COLUMN, MDC_CORRELATION_ID_COLUMN
        ).mapInPandas(
            tranform_df_function_with_correlation_id, schema=data_as_df.schema
        )
    else:
        transformed_df = df.select(MDC_DATA_COLUMN).mapInPandas(
            transform_df_function_without_correlation_id, schema=data_as_df.schema
        )

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
