# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal logic for Trace Aggregator step of Gen AI preprocessor component."""


import json
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


def _validate_traces_df(aggregated_traces_df: DataFrame) -> DataFrame:
    """Check specific validations for aggregated trace entries as necessary."""
    # some PF logs that result in exception won't have output so need to drop traces that don't have output or input.
    transformed_df = aggregated_traces_df.dropna(subset=["output", "input"])

    _count_dropped_rows_with_error(
        aggregated_traces_df.count(),
        transformed_df.count(),
        additional_error_msg="Additionally, the step that caused issues was validating trace logs input/output." + \
            " Double check the stdout and spark executor logs for debug info to find root cause of issue.")

    return transformed_df


def _aggregate_span_logs_to_trace_logs(grouped_row: Row):
    """Aggregate grouped span logs into trace logs."""
    output_schema = _get_aggregated_trace_log_spark_df_schema()

    span_list = [SpanTreeNode(row) for row in grouped_row.span_rows]
    tree = SpanTree(span_list)
    if tree.root_span is None:
        # until we replace trace_id with request_id for PF logs we will have multiple root_span for each
        # trace_id and so we have to split them up manually.
        seperated_trace_entries = []

        for (root_span, trace_idx) in zip(tree.possible_root_spans, range(len(tree.possible_root_spans))):
            root_span.trace_id = f"{root_span.trace_id}_{trace_idx}"

            output_dict = root_span.to_dict(datetime_to_str=False)
            output_dict['input'] = root_span.input
            output_dict['output'] = root_span.output
            output_dict['root_span'] = json.dumps(root_span.to_dict())

            seperated_trace_entries.append(
                tuple(output_dict.get(fieldName, None) for fieldName in output_schema.fieldNames())
            )
        return seperated_trace_entries
    else:
        output_dict = tree.root_span.to_dict(datetime_to_str=False)
        output_dict['input'] = tree.root_span.input
        output_dict['output'] = tree.root_span.output
        output_dict['root_span'] = tree.to_json_str()

        return [tuple(output_dict.get(fieldName, None) for fieldName in output_schema.fieldNames())]


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
    output_trace_schema = _get_aggregated_trace_log_spark_df_schema()

    if not require_trace_data:
        spark = init_spark()
        print("Skip processing of spans into aggregated traces.")
        return spark.createDataFrame(data=[], schema=output_trace_schema)

    print("Processing spans into aggregated traces...")

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

    all_aggregated_traces = grouped_spans_df \
        .rdd \
        .flatMap(_aggregate_span_logs_to_trace_logs) \
        .toDF(output_trace_schema)

    all_aggregated_traces = _filter_df_by_time_window(
        all_aggregated_traces, data_window_start, data_window_end)

    print("Aggregated Trace DF:")
    all_aggregated_traces.show()
    all_aggregated_traces.printSchema()

    validated_traces = _validate_traces_df(all_aggregated_traces)

    return validated_traces
