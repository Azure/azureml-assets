# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Model Data Collector Data Window Component."""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.functions import posexplode, concat_ws
from dateutil import parser
from datetime import datetime
from mdc_preprocessor_helper import get_datastore_name_from_input_path
from shared_utilities.momo_exceptions import DataNotFoundError
from shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
)
from shared_utilities.event_utils import add_tags_to_root_run
from shared_utilities.constants import (
    MDC_CORRELATION_ID_COLUMN,
    MDC_DATA_COLUMN,
    MDC_DATAREF_COLUMN,
    AML_MOMO_ERROR_TAG
)
from mdc_preprocessor_helper import get_file_list


def _mdc_uri_folder_to_raw_spark_df(start_datetime: datetime, end_datetime: datetime, input_data: str,
                                    add_tags_func=None) -> DataFrame:
    def handle_data_not_found():
        add_tags_func({AML_MOMO_ERROR_TAG: "No data found for the given time window."})
        raise DataNotFoundError(f"No data found for the given time window: {start_datetime} to {end_datetime}")

    add_tags_func = add_tags_func or add_tags_to_root_run
    file_list = get_file_list(start_datetime, end_datetime, input_data)
    if not file_list:
        handle_data_not_found()
    # print("DEBUG file_list:", file_list)

    spark = init_spark()
    df = spark.read.json(file_list)
    if df.rdd.isEmpty():
        handle_data_not_found()
    return df


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

    columns = _get_data_columns(df)

    # TODO handle dataref
    if extract_correlation_id:
        columns.append(MDC_CORRELATION_ID_COLUMN)
        # explode the data column of array type to multiple rows with index
        df = df[columns].select(posexplode(MDC_DATA_COLUMN).alias("index", "value"), MDC_CORRELATION_ID_COLUMN)
        # set the new correlationid as {correlationid}_{index}
        df = df.withColumn(MDC_CORRELATION_ID_COLUMN, concat_ws("_", MDC_CORRELATION_ID_COLUMN, "index")).drop("index")
        # select the 1st level features as columns
        df = df.select("value.*", MDC_CORRELATION_ID_COLUMN)
    else:
        df = df[columns].select(posexplode(MDC_DATA_COLUMN).alias("index", "value"))
        df = df.select("value.*")
    return df


def _mdc_uri_folder_to_preprocessed_spark_df(
        data_window_start: datetime, data_window_end: datetime, input_data: str, extract_correlation_id: bool,
        add_tags_func=None) -> DataFrame:
    """Read raw MDC data, preprocess, and return in a Spark DataFrame."""
    # Parse the dates
    start_datetime = parser.parse(data_window_start)
    end_datetime = parser.parse(data_window_end)

    df = _mdc_uri_folder_to_raw_spark_df(start_datetime, end_datetime, input_data, add_tags_func)
    print("df converted from MDC raw uri folder:")
    df.select("data").show(truncate=False)
    df.printSchema()

    datastore = get_datastore_name_from_input_path(input_data)
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
        extract_correlation_id: bool):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_data: The mltable path pointing to location where the outputted mltable will be written to.
        extract_correlation_id: The boolean to extract correlation Id from the MDC logs.
    """
    transformed_df = _mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, input_data,
                                                              extract_correlation_id)

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
