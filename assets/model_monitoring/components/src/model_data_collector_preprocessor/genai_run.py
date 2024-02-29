# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for GenAI MDC Preprocessor."""

import argparse

from datetime import datetime, timedelta
from dateutil import parser
from pyspark.sql import DataFrame
from pyspark.sql.types import TimestampType
from pyspark.sql.utils import AnalysisException
from pyspark.sql.functions import lit
from shared_utilities.momo_exceptions import InvalidInputError
from shared_utilities.io_utils import save_spark_df_as_mltable
from model_data_collector_preprocessor.store_url import StoreUrl

from model_data_collector_preprocessor.mdc_utils import (
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


def try_get_df_column(df: DataFrame, name: str):
    """Get column if it exists in DF. Return none if column does not exist."""
    try:
        return df[name]
    except AnalysisException:
        return None


def _drop_promoted_fields(df: DataFrame, promoted_fields_mapping: dict) -> DataFrame:
    """Drop the promoted fields from dataframe to avoid data duplication and save storage space."""
    def try_drop_field(df: DataFrame, col_name: str, field_name: str):
        """Drop field from nested columns like context, attributes.
        Will set column to null if dropping last field in the column.
        No op if encounter exception.
        """
        df_col = try_get_df_column(df, col_name)
        if df_col is None:
            return df
        try:
            return df.withColumn(col_name, df_col.dropFields(field_name))
        except AnalysisException as ex:
            if "DATATYPE_MISMATCH.CANNOT_DROP_ALL_FIELDS" in str(ex):
                return df.withColumn(col_name, lit(None))
            print("MoMo internal exception encountered: \n", ex)
            return df

    # remove promoted data fields from source to avoid duplication. 'status' col is already overwritten.
    promoted_fields_mapping.pop('status', None)

    for field_name_in_data in promoted_fields_mapping.values():
        col_name, nested_field_name = field_name_in_data.split('.')
        df = try_drop_field(df, col_name, nested_field_name)

    return df


def _promote_fields_from_attributes(df: DataFrame) -> DataFrame:
    """Retrieve common/important fields(span_type, input, etc.) from attributes and make them first level columns.

    Args:
        df: The raw span logs data in a dataframe.
    """
    fields_to_promote_mapping = _get_important_field_mapping()
    df = df.withColumns(
        {
            key: (df_col if (df_col := try_get_df_column(df, col_name)) is not None else lit(df_col))
            for key, col_name in fields_to_promote_mapping.items()
        }
    )

    # as of right now UX does not want us to remove the promoted fields. Uncomment logic if we need to change it later.
    # df = _drop_promoted_fields(df, fields_to_promote_mapping)
    return df


def _preprocess_raw_logs_to_span_logs_spark_df(df: DataFrame) -> DataFrame:
    """Apply Gen AI preprocessing steps to raw logs dataframe.

    Args:
        df: The raw span logs data in a dataframe.
    """
    df = df.withColumns(
        {'end_time': df.end_time.cast(TimestampType()), 'start_time': df.start_time.cast(TimestampType())}
    )

    # check that cast was successful. Failed cast will result in 'null' values
    if not df.filter(df.start_time.isNull()).isEmpty() or not df.filter(df.end_time.isNull()).isEmpty():
        raise InvalidInputError(
            "The start or end time columns of the raw span logs contain invalid Timestamp strings." +
            " The strings should be parseable by pyspark's TimestampType(), usually we expect iso-format." +
            " Double check the raw input data 'start_time' and 'end_time' column values."
        )

    df = _promote_fields_from_attributes(df)

    # Cast all remaining fields in attributes to json string, for easier schema unifying
    df = _convert_complex_columns_to_json_string(df)

    return df


def _filter_look_backward_logs(logs_df: DataFrame, data_window_start: datetime) -> DataFrame:
    """Filter out logs that were got from our 1-hour look backward window but are not in our desired time window."""
    desired_trace_ids = logs_df.where(logs_df.start_time >= data_window_start).select('trace_id').distinct()
    filtered_logs = desired_trace_ids.join(
        logs_df, on="trace_id", how="left"
    )
    return filtered_logs


def _genai_uri_folder_to_preprocessed_spark_df(
        data_window_start: str, data_window_end: str, store_url: StoreUrl, add_tags_func=None
) -> DataFrame:
    """Read raw gen AI logs data, preprocess, and return in a Spark DataFrame."""
    # look-back 1 hour for extra span logs
    data_window_start_as_datetime = parser.parse(data_window_start)
    adjusted_date_window_start = data_window_start_as_datetime - timedelta(hours=1)

    df = _mdc_uri_folder_to_preprocessed_spark_df(
        adjusted_date_window_start.strftime("%Y%m%dT%H:%M:%S"), data_window_end, store_url, False, add_tags_func)

    df = _preprocess_raw_logs_to_span_logs_spark_df(df)

    df = _filter_look_backward_logs(df, data_window_start_as_datetime)

    print("df processed from raw Gen AI logs:")
    df.show()
    df.printSchema()

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

    span_logs_df = _genai_uri_folder_to_preprocessed_spark_df(data_window_start, data_window_end, store_url)

    trace_logs_df = process_spans_into_aggregated_traces(span_logs_df, require_trace_data)

    save_spark_df_as_mltable(span_logs_df, preprocessed_span_data)

    save_spark_df_as_mltable(trace_logs_df, aggregated_trace_data)


def run():
    """Compute data window and preprocess data from MDC."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--data_window_start", type=str)
    arg_parser.add_argument("--data_window_end", type=str)
    arg_parser.add_argument("--input_data", type=str)
    arg_parser.add_argument("--preprocessed_span_data", type=str)
    arg_parser.add_argument("--aggregated_trace_data", type=str)
    arg_parser.add_argument("--require_trace_data", type=bool, default=True)
    args = arg_parser.parse_args()

    genai_preprocessor(
        args.data_window_start,
        args.data_window_end,
        args.input_data,
        args.preprocessed_span_data,
        args.aggregated_trace_data,
        args.require_trace_data,
    )


if __name__ == "__main__":
    run()
