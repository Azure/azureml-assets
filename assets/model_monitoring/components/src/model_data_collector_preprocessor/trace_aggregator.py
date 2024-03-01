# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal logic for Trace Aggregator step of Gen AI preprocessor component."""


from pyspark.sql import DataFrame
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import TimestampType, StructType, StructField, StringType
from shared_utilities.span_tree_utils import SpanTree, SpanTreeNode
from model_data_collector_preprocessor.genai_preprocessor_df_schemas import (
    _get_aggregated_trace_log_spark_df_schema,
)
from shared_utilities.io_utils import init_spark


def _aggregate_span_logs_to_trace_logs(grouped_row):
    """Aggregate grouped span logs into trace logs."""
    output_schema = StructType(
        [
            StructField("trace_id", StringType(), True),
            StructField("user_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", TimestampType(), True),
            StructField("end_time", TimestampType(), True),
            StructField("input", StringType(), True),
            StructField("output", StringType(), True),
            StructField("root_span", StringType(), True),
        ]
    )

    span_list = [SpanTreeNode(row) for row in grouped_row.span_rows]
    tree = SpanTree(span_list)
    if tree.root_span is None:
        output_dict = {}
    else:
        output_dict = tree.root_span.to_dict(datetime_to_str=False)
        output_dict['root_span'] = tree.to_json_str()
    return tuple(output_dict.get(fieldName, None) for fieldName in output_schema.fieldNames())


def process_spans_into_aggregated_traces(span_logs: DataFrame, require_trace_data: bool) -> DataFrame:
    """Group span logs into aggregated trace logs."""
    spark = init_spark()
    output_trace_schema = _get_aggregated_trace_log_spark_df_schema()

    if not require_trace_data:
        print("Skip processing of spans into aggregated traces.")
        return spark.createDataFrame(data=[], schema=output_trace_schema)

    print("Processing spans into aggregated traces...")

    grouped_spans_df = span_logs.groupBy('trace_id').agg(
        collect_list(
            struct(span_logs.schema.fieldNames())
        ).alias('span_rows')
    )

    # need to allow null values just in case of empty rdd entry from faulty SpanTree construction.
    for field in output_trace_schema.fields:
        field.nullable = True

    all_aggregated_traces = grouped_spans_df \
        .rdd \
        .map(_aggregate_span_logs_to_trace_logs) \
        .toDF(output_trace_schema)

    # remove any null root_span rows
    all_aggregated_traces = all_aggregated_traces.dropna(how="all")

    print("Aggregated Trace DF:")
    all_aggregated_traces.show()
    all_aggregated_traces.printSchema()
    return all_aggregated_traces
