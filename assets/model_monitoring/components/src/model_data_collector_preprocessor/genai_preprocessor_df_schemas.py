# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Constants file for genai preprocessor component."""


from pyspark.sql.types import TimestampType, StructType, StructField, StringType


def _get_preprocessed_span_logs_df_schema() -> StructType:
    """Get processed span logs Dataframe schema."""
    # TODO: The user_id and session_id may not be available in v1.
    schema = StructType([
        StructField('attributes', StringType(), False),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('input', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('output', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
    ])
    return schema


def _get_aggregated_trace_log_spark_df_schema() -> StructType:
    """Get Aggregated Trace Log DataFrame Schema."""
    # TODO: The user_id and session_id may not be available in v0 of trace aggregator.
    schema = StructType(
        [
            StructField("end_time", TimestampType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
            # StructField("session_id", StringType(), True),
            StructField("start_time", TimestampType(), False),
            StructField("trace_id", StringType(), False),
            # StructField("user_id", StringType(), True),
        ]
    )
    return schema
