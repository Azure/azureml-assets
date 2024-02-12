# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GenAI MDC Preprocessor."""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.functions import posexplode, concat_ws, udf, from_json, when, col, to_json
from pyspark.sql.types import TimestampType
from shared_utilities.momo_exceptions import DataNotFoundError
from shared_utilities.io_utils import init_spark, save_spark_df_as_mltable
from model_data_collector_preprocessor.store_url import StoreUrl
from model_data_collector_preprocessor.spark_run import (
    _mdc_uri_folder_to_preprocessed_spark_df,
    _convert_complex_columns_to_json_string,
)


def _preprocess_raw_logs_in_spark_df(df: DataFrame) -> DataFrame:
    """Apply Gen AI preprocessing steps to raw logs dataframe.
    
    Args:
        df: The raw span logs data in a dataframe.
    """
    df.withColumns(
        {'end_time': df.end_time.cast(TimestampType()), 'start_time': df.start_time.cast(TimestampType())}
    )

    # Retrieve common/important fields(span_type, input, output, etc.) from attributes and make them first level columns
    
    # Cast all remaining fields in attributes to string to make it Map(String, String), or make it json string, for easier schema unifying
    return df


def genai_preprocessor(
        data_window_start: str,
        data_window_end: str,
        input_data: str,
        preprocessed_span_data: str,
        aggregated_trace_data: str):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_span_data: The mltable path pointing to location where the outputted span logs mltable will be written to.
        aggregated_trace_data: The mltable path pointing to location where the outputted aggregated trace logs mltable will be written to.
    """
    store_url = StoreUrl(input_data)

    transformed_df = _mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url, extract_correlation_id=False)

    transformed_df = _preprocess_raw_logs_in_spark_df(transformed_df)

    # TODO: remove this step after we switch our interface from mltable to uri_folder
    transformed_df = _convert_complex_columns_to_json_string(transformed_df)

    save_spark_df_as_mltable(transformed_df, preprocessed_span_data)
    save_spark_df_as_mltable(transformed_df, aggregated_trace_data)

def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--preprocessed_span_data", type=str)
    parser.add_argument("--aggregated_trace_data", type=str)
    args = parser.parse_args()

    genai_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_span_data,
        args.aggregated_trace_data,
    )


if __name__ == "__main__":
    run()