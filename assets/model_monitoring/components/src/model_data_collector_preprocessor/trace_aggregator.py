# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Internal logic for Trace Aggregator step of Gen AI preprocessor component."""


from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from typing import List

from shared_utilities.io_utils import init_spark
from model_data_collector_preprocessor.span_tree_utils import SpanTree, SpanTreeNode


def _get_aggregated_trace_log_spark_df_schema() -> StructType:
    """Get Aggregated Trace Log DataFrame Schema."""
    # TODO: The user_id and session_id may not be available in v0 of trace aggregator.
    schema = StructType(
        [
            StructField("end_time", TimestampType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", TimestampType(), False),
            StructField("trace_id", StringType(), False),
            StructField("user_id", StringType(), True),
        ]
    )
    return schema


def _construct_aggregated_trace_df(span_tree: SpanTree) -> DataFrame:
    """Build an aggregated trace dataframe from a span tree."""
    spark = init_spark()
    trace_schema = _get_aggregated_trace_log_spark_df_schema()
    agg_trace_schema_names = trace_schema.fieldNames()
    span_dict = span_tree.root_span.span_row.asDict()

    data = {key_name: span_dict.get(key_name, None) for key_name in agg_trace_schema_names}
    data['root_span'] = span_tree.to_json_str()

    return spark.createDataFrame([data], trace_schema)  # type: ignore


def _construct_span_tree(span_rows: List[Row]) -> SpanTree:
    """Build a span tree from the raw span rows."""
    span_list = [SpanTreeNode(row) for row in span_rows]
    tree = SpanTree(span_list)
    return tree


def process_spans_into_aggregated_traces(span_logs: DataFrame) -> DataFrame:
    """Group span logs into aggregated trace logs."""
    spark = init_spark()
    distinct_trace_ids = span_logs.select("trace_id").distinct()
    distinct_trace_ids.show()

    all_aggregated_traces = spark.createDataFrame(data=[], schema=_get_aggregated_trace_log_spark_df_schema())
    for trace_id in distinct_trace_ids.collect():
        grouped_spans_df = span_logs.where(span_logs.trace_id == trace_id.trace_id)
        tree = _construct_span_tree(grouped_spans_df.collect())
        new_entry = _construct_aggregated_trace_df(tree)
        all_aggregated_traces = all_aggregated_traces.union(new_entry)

    all_aggregated_traces.show()
    return all_aggregated_traces
