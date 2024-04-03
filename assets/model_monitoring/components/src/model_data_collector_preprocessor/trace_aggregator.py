# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal logic for Trace Aggregator step of Gen AI preprocessor component."""


from pyspark.sql import DataFrame
from datetime import datetime
from pyspark.sql.functions import collect_list, struct
from shared_utilities.span_tree_utils import SpanTree, SpanTreeNode
from model_data_collector_preprocessor.mdc_utils import _filter_df_by_time_window
from model_data_collector_preprocessor.genai_preprocessor_df_schemas import (
    _get_aggregated_trace_log_spark_df_schema,
)
from shared_utilities.io_utils import init_spark


def _aggregate_span_logs_to_trace_logs(grouped_row):
    """Aggregate grouped span logs into trace logs."""
    output_schema = _get_aggregated_trace_log_spark_df_schema()

    span_list = [SpanTreeNode(row) for row in grouped_row.span_rows]
    tree = SpanTree(span_list)
    if tree.root_span is None:
        return []
    else:
        output_dict = tree.root_span.to_dict(datetime_to_str=False)
        output_dict['input'] = tree.root_span.input
        output_dict['output'] = tree.root_span.output
        output_dict['root_span'] = tree.to_json_str()
        return [tuple(output_dict.get(fieldName, None) for fieldName in output_schema.fieldNames())]


def aggregate_spans_into_traces(
        enlarged_span_logs: DataFrame, require_trace_data: bool,
        data_window_start: datetime, data_window_end: datetime) -> DataFrame:
    """Group span logs into aggregated trace logs."""
    output_trace_schema = _get_aggregated_trace_log_spark_df_schema()

    # TODO: figure out optional output behavior and change to that.
    if not require_trace_data:
        spark = init_spark()
        print("Skip processing of spans into aggregated traces.")
        return spark.createDataFrame(data=[], schema=output_trace_schema)

    print("Processing spans into aggregated traces...")

    # TODO: change this conditional to check against real schema once we know what the new log schema will be
    if "request_id" in enlarged_span_logs.columns:
        print("Found 'request_id' in schema, most likely PromptFlow raw logs. Grouping by 'request_id'...")
        enlarged_span_logs = enlarged_span_logs.drop('trace_id').withColumn('trace_id', enlarged_span_logs.request_id)
    else:
        print("Found no 'request_id' in schema. Skip promoting as 'trace_id'.")

    grouped_spans_df = enlarged_span_logs.groupBy('trace_id').agg(
        collect_list(
            struct(enlarged_span_logs.schema.fieldNames())
        ).alias('span_rows')
    )

    all_aggregated_traces = grouped_spans_df \
        .rdd \
        .flatMap(_aggregate_span_logs_to_trace_logs) \
        .toDF(output_trace_schema)

    all_aggregated_traces = _filter_df_by_time_window(
        all_aggregated_traces, data_window_start, data_window_end)

    print("Aggregated Trace DF:")
    all_aggregated_traces.show()
    all_aggregated_traces.printSchema()
    return all_aggregated_traces
