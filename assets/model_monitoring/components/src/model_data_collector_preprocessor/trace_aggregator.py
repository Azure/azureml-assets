# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bisect
import json
from math import dist

from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import from_json, to_json, udf, collect_list
from typing import Dict, Iterator, List, Union

from assets.model_monitoring.components.src.shared_utilities.io_utils import init_spark


class SpanTreeNode:
    def __init__(self, span_row: Row):
        self.span_row = span_row
        self._children: List[SpanTreeNode] = []

    @property
    def span_id(self) -> str:
        """Get the span id."""
        return self.span_row.span_id

    @property
    def parent_id(self) -> str:
        """Get the span's parent id."""
        return self.span_row.parent_id

    @property
    def children(self) -> List["SpanTreeNode"]:
        """Get the span's children as list."""
        return self._children

    def insert_child(self, span):
        """Inserts a child span in ascending time order due to __lt__()."""
        bisect.insort(self._children, span)

    def show(self, indent=0):
        print(f"{' '*indent}[{self.span_row.span_id}({self.span_row.start_time}, {self.span_row.end_time})]")
        for c in self.children:
            c.show(indent+4)

    def __iter__(self) -> Iterator["SpanTreeNode"]:
        for child_span in self._children:
            for span in child_span:
                yield span
        yield self

    def __lt__(self, other):
        """Custom less-than comparison for sorting by time in bisect.insort() for python3.8."""
        return self.span_row.end_time < other.span_row.end_time

    def to_dict(self) -> dict:
        """Dictionary representation of Span."""
        span_node_schema_names = _get_span_tree_node_spark_df_schema().fieldNames()
        span_dict = self.span_row.asDict()
        out_dict = {key_name: span_dict.get(key_name) for key_name in span_node_schema_names}
        out_dict['children'] = self.children
        return out_dict


class SpanTree:
    def __init__(self, spans: List[SpanTreeNode]) -> None:
        self.root_span = self._construct_span_tree(spans)

    def _construct_span_tree(self, spans: List[SpanTreeNode]):
        # construct a dict with span_id as key and span as value
        span_map = { span.span_id: span for span in spans }
        for span in span_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = span_map.get(parent_id)
                if parent_span is not None:
                    parent_span.insert_child(span)
        return root_span

    def show(self):
        if self.root_span is None:
            return
        self.root_span.show()

    def __iter__(self) -> Iterator[SpanTreeNode]:
        if self.root_span is None:
            return
        for span in self.root_span.__iter__():
            yield span

    def to_json_str(self) -> str:
        """Function to return jsons tring tree structure."""
        # TODO:
        output_dict = {}
        self._get_json_str_repr(self.root_span, output_dict)
        return json.dumps(output_dict['root_span'])

    def _get_json_str_repr(self, curr_span: SpanTreeNode, output: dict) -> str:
        """Recursively get tree structure JSON string."""
        # TODO:
        for child in curr_span.children:
            self._get_json_str_repr(child, output)

def _get_span_tree_node_spark_df_schema():
    """Get SpanTree spark df schema."""
    schema = StructType(
        [
            StructField("parent_id", StringType(), True),
            StructField("span_id", StringType(), False),
            StructField("span_type", StringType(), False),
            StructField("start_time", StringType(), False),
            StructField("end_time", StringType(), False),
            StructField("children", ArrayType(StringType(), True), False),
        ]
    )
    return schema

def _get_aggregated_trace_log_spark_df_schema():
    """Get Aggregated Trace Log DataFrame Schema."""
    # TODO: The user_id and session_id may not be available in v0 of trace aggregator.
    schema = StructType(
        [
            StructField("trace_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", StringType(), False),
            StructField("end_time", StringType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
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
    # TODO: decide jsonString format for tree and encode below:
    data['root_span'] = str(span_tree)

    return spark.createDataFrame([(data)], trace_schema)


def _construct_span_tree(span_rows: List[Row]) -> SpanTree:
    """Build a span tree from the raw span rows."""
    span_list = [SpanTreeNode(row) for row in span_rows]
    tree = SpanTree(span_list)
    return tree


def process_spans_into_aggregated_traces(span_logs: DataFrame) -> DataFrame:
    """Group span logs into aggregated trace logs."""
    spark = init_spark()
    distinct_trace_ids = span_logs.select("trace_id").distinct()
    all_aggregated_traces = spark.createDataFrame(data=[], schema=_get_aggregated_trace_log_spark_df_schema())
    for trace_id in distinct_trace_ids.collect():
        tree = _construct_span_tree(span_logs.where(span_logs.trace_id == trace_id.trace_id).collect())
        new_entry = _construct_aggregated_trace_df(tree)
        all_aggregated_traces.union(new_entry)
    return all_aggregated_traces
