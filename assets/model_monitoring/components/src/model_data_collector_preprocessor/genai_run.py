# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GenAI MDC Preprocessor."""

import argparse

from pyspark.sql import DataFrame
from pyspark.sql.types import TimestampType
from pyspark.sql.utils import AnalysisException
from shared_utilities.io_utils import save_spark_df_as_mltable
from model_data_collector_preprocessor.store_url import StoreUrl
# TODO: move shared utils to a util py file
from model_data_collector_preprocessor.spark_run import (
    _mdc_uri_folder_to_preprocessed_spark_df,
    _convert_complex_columns_to_json_string,
)
from model_data_collector_preprocessor.trace_aggregator import (
    process_spans_into_aggregated_traces,
)


def _get_important_field_mapping() -> dict:
    """Map the span log schema names to the expected raw log schema column/field name."""
    map = {
        "trace_id": "context.trace_id",
        "span_id": "context.span_id",
        "span_type": "attributes.span_type",
        "status": "status.status_code",
        "framework": "attributes.framework",
        "input": "attributes.inputs",
        "output": "attributes.output",
    }
    return map


def _promote_fields_from_attributes(df: DataFrame) -> DataFrame:
    """Retrieve common/important fields(span_type, input, etc.) from attributes and make them first level columns.

    Args:
        df: The raw span logs data in a dataframe.
    """
    def try_get_df_column(df: DataFrame, name: str):
        try:
            return df[name]
        except AnalysisException:
            return None

    df = df.withColumns(
        {
            key: df_column for key, col_name in _get_important_field_mapping().items()
            if (df_column := try_get_df_column(df, col_name)) is not None
        }
    )
    return df


def _preprocess_raw_logs_to_span_logs_spark_df(df: DataFrame) -> DataFrame:
    """Apply Gen AI preprocessing steps to raw logs dataframe.

    Args:
        df: The raw span logs data in a dataframe.
    """
    # TODO: handle if original start/endtime not valid time string
    df = df.withColumns(
        {'end_time': df.end_time.cast(TimestampType()), 'start_time': df.start_time.cast(TimestampType())}
    )

    df = _promote_fields_from_attributes(df)

    # Cast all remaining fields in attributes to json string, for easier schema unifying
    df = _convert_complex_columns_to_json_string(df)

    print("df processed from raw Gen AI logs:")
    df.show(truncate=False)
    df.printSchema()

    return df


def _genai_uri_folder_to_preprocessed_spark_df(
        data_window_start: str, data_window_end: str, store_url: StoreUrl, add_tags_func=None
) -> DataFrame:
    """Read raw gen AI logs data, preprocess, and return in a Spark DataFrame."""
    df = _mdc_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url, False, add_tags_func)

    df = _preprocess_raw_logs_to_span_logs_spark_df(df)

    return df


def genai_preprocessor(
        data_window_start: str,
        data_window_end: str,
        input_data: str,
        preprocessed_span_data: str,
        aggregated_trace_data: str,
        require_trace_data: bool):
    """Extract data based on window size provided and preprocess it into MLTable.

    Args:
        data_window_start: The start date of the data window.
        data_window_end: The end date of the data window.
        input_data: The data asset on which the date filter is applied.
        preprocessed_span_data: The mltable path pointing to location where the
        outputted span logs mltable will be written to.
        aggregated_trace_data: The mltable path pointing to location where the
        outputted aggregated trace logs mltable will be written to.
    """
    store_url = StoreUrl(input_data)

    transformed_df = _genai_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url)

    trace_logs_df = process_spans_into_aggregated_traces(transformed_df, require_trace_data)

    save_spark_df_as_mltable(transformed_df, preprocessed_span_data)

    save_spark_df_as_mltable(trace_logs_df, aggregated_trace_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_window_start", type=str)
    parser.add_argument("--data_window_end", type=str)
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--preprocessed_span_data", type=str)
    parser.add_argument("--aggregated_trace_data", type=str)
    parser.add_argument("--require_trace_data", type=str)
    args = parser.parse_args()

    genai_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_span_data,
        args.aggregated_trace_data,
        eval(args.require_trace_data.capitalize())
    )


if __name__ == "__main__":
    run()
