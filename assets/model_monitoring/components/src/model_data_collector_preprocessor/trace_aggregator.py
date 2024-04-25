# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal logic for Trace Aggregator step of Gen AI preprocessor component."""


import json
from pyspark import Accumulator
from pyspark.sql import DataFrame, Row
from datetime import datetime
from pyspark.sql.functions import collect_list, struct
from shared_utilities.span_tree_utils import SpanTree, SpanTreeNode
from model_data_collector_preprocessor.mdc_utils import (
    _filter_df_by_time_window,
    _count_dropped_rows_with_error,
)
from model_data_collector_preprocessor.genai_preprocessor_df_schemas import (
    _get_aggregated_trace_log_spark_df_schema,
)
from shared_utilities.io_utils import init_spark


logging_prefix_str = "[TRACE_AGGREGATOR_LOGS] [INFO]"


def _create_trace_log_output_entry(
        output_dict: dict, error_rate_accumulator: Accumulator[int]):
    """Validate and create an entry for trace log output DF from output_dict."""
    output_schema = _get_aggregated_trace_log_spark_df_schema()

    if output_dict.get('input', None) is None:
        error_rate_accumulator.add(1)
        print(f"{logging_prefix_str} 'Input' for trace = {output_dict.get('trace_id', None)} and"
              f" root_span id = {output_dict.get('span_id', None)} is null."
              " Discarding trace entry.")
        return None
    if output_dict.get('output', None) is None:
        error_rate_accumulator.add(1)
        print(f"{logging_prefix_str} 'Output' for trace = {output_dict.get('trace_id', None)} and"
              f" root_span id = {output_dict.get('span_id', None)} is null."
              " Discarding trace entry.")
        return None
    return tuple(output_dict.get(fieldName, None) for fieldName in output_schema.fieldNames())


def _aggregate_span_logs_to_trace_logs(grouped_row: Row, error_rate_accumulator: Accumulator[int]):
    """Aggregate grouped span logs into trace logs."""
    span_list = [SpanTreeNode(row) for row in grouped_row.span_rows]
    tree = SpanTree(span_list)

    if tree.root_span is None:
        print(f"{logging_prefix_str} Didn't find a singlular root_span."
              "Attemting to flatten forest of root_spans if possible...")
        # until we replace trace_id with request_id for PF logs we will have multiple root_span for each
        # trace_id and so we have to split them up manually.
        seperated_trace_entries = []

        for (root_span, trace_idx) in zip(tree.possible_root_spans, range(len(tree.possible_root_spans))):
            root_span.trace_id = f"{root_span.trace_id}_{trace_idx}"

            output_dict = root_span.to_dict(datetime_to_str=False)
            output_dict['input'] = root_span.input
            output_dict['output'] = root_span.output
            output_dict['root_span'] = json.dumps(root_span.to_dict())

            entry = _create_trace_log_output_entry(output_dict, error_rate_accumulator)
            if entry is not None:
                seperated_trace_entries.append(entry)

        print(f"{logging_prefix_str} Finish flattening forest.")
        return seperated_trace_entries
    else:
        print(f"{logging_prefix_str} Found a singlular root_span. Creating entry for it.")
        output_dict = tree.root_span.to_dict(datetime_to_str=False)
        output_dict['input'] = tree.root_span.input
        output_dict['output'] = tree.root_span.output
        output_dict['root_span'] = tree.to_json_str()

        entry = _create_trace_log_output_entry(output_dict, error_rate_accumulator)
        return [] if entry is None else [entry]


def _replace_trace_with_request_id(row: Row):
    """Replace the trace_id value with request_id if request_id is not null."""
    output_dict: dict = row.asDict()
    if 'attributes' in output_dict:
        attributes_dict: dict = json.loads(output_dict['attributes'])
        if attributes_dict is not None and 'request_id' in attributes_dict \
                and attributes_dict.get('request_id', None) is not None:
            output_dict['trace_id'] = attributes_dict.get('request_id')
            return Row(**output_dict)
    return row


def aggregate_spans_into_traces(
        enlarged_span_logs: DataFrame, require_trace_data: bool,
        data_window_start: datetime, data_window_end: datetime) -> DataFrame:
    """Group span logs into aggregated trace logs."""
    spark = init_spark()
    output_trace_schema = _get_aggregated_trace_log_spark_df_schema()

    if not require_trace_data:
        print(f"{logging_prefix_str} Skip processing of spans into aggregated traces.")
        return spark.createDataFrame(data=[], schema=output_trace_schema)

    print(f"{logging_prefix_str} Processing spans into aggregated traces...")

    # for PromptFlow we need to replace the trace_id values with request_id in order to handle edge cases
    # where PF has same trace_id but multiple LLM requests
    # TODO: Prompt flow team doesn't guarantee that all spans will have request_id due to a backlog of work.
    # Uncomment the code below when we get a guarantee from them
    # enlarged_span_logs = enlarged_span_logs.rdd.map(_replace_trace_with_request_id).toDF(enlarged_span_logs.schema)

    grouped_spans_df = enlarged_span_logs.groupBy('trace_id').agg(
        collect_list(
            struct(enlarged_span_logs.schema.fieldNames())
        ).alias('span_rows')
    )

    error_rate_accumulator = spark.sparkContext.accumulator(0)

    all_aggregated_traces = grouped_spans_df \
        .rdd \
        .flatMap(lambda row: _aggregate_span_logs_to_trace_logs(row, error_rate_accumulator)) \
        .toDF(output_trace_schema)

    all_aggregated_traces = _filter_df_by_time_window(
        all_aggregated_traces, data_window_start, data_window_end)

    print(f"{logging_prefix_str} Aggregated Trace DF:")
    all_aggregated_traces.show()
    all_aggregated_traces.printSchema()

    # check how many rows were dropped from the final trace output and check against threshold.
    final_trace_row_count = all_aggregated_traces.count()
    _count_dropped_rows_with_error(
        final_trace_row_count + error_rate_accumulator.value,
        final_trace_row_count,
        additional_error_msg="Additionally, the most likely cause for dropped rows is trace logs with missing"
        " input/output. Double check the spark executor logs for debug info to find root cause of issue.")

    return all_aggregated_traces
