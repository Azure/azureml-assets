# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import bisect
import json
from math import dist

from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StructField, StringType, TimestampNTZType
from pyspark.sql.functions import from_json, to_json, udf, collect_list
from typing import Dict, Iterator, List, Union

from assets.model_monitoring.components.src.shared_utilities.io_utils import init_spark

        

class Span:
    def __init__(self, span_row: Row):
        self.span_row = span_row
        self.children = []

    def show(self, indent=0):
        print(f"{' '*indent}[{self.span_row.span_id}({self.span_row.start_time}, {self.span_row.end_time})]")
        for c in self.children:
            c.show(indent+4)

    @property
    def span_id(self) -> str:
        """Get the span id."""
        return self.span_row.span_id

    @property
    def parent_id(self) -> str:
        """Get the span's parent id."""
        return self.span_row.parent_id

    def __iter__(self) -> Iterator["Span"]:
        for child_span in self.children:
            for span in child_span:
                yield span
        yield self

    def from_dict(self, row_dict: dict):
        """Load Span object from a pyspark.sql.Row.AsDict() representation."""
        self.span_id = row_dict.get("trace_id")


class SpanTree:
    def __init__(self, spans: List[Span]) -> None:
        self.root_span = self._construct_span_tree(spans)

    def _construct_span_tree(self, spans: List[Span]):
        # construct a dict with span_id as key and span as value
        span_map = { span.span_id: span for span in spans }
        for span in span_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = span_map.get(parent_id)
                # insert in order of end time
                bisect.insort(parent_span.children, span, key = lambda s: s.end)
        return root_span

    def show(self):
        if self.root_span is None:
            return
        self.root_span.show()

    def __iter__(self) -> Iterator[Span]:
        for span in self.root_span.__iter__():
            yield span

    def to_json_string(self) -> str:
        return SpanTreeSerialization(self.root_span).to_json_str()


class SpanTreeSerialization():
    def __init__(self, span_tree: SpanTree):
        self.tree = span_tree

    def to_json_str(self) -> str:
        
        for span in self.tree:
            
        return ""


def _get_aggregated_trace_log_spark_df_schema():
    """Get Aggregated Trace Log DataFrame Schema."""
    # TODO: The user_id and session_id may not be available in v0 of trace aggregator.
    schema = StructType(
        [
            StructField("trace_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", TimestampNTZType(), False),
            StructField("end_time", TimestampNTZType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
        ]
    )
    return schema


def _construct_span_tree(span_rows: List[Row]):
    """Transforms a single trace id into a """
    span_list = [Span(row) for row in span_rows]
    tree_builder = SpanTree(spans)
    tree_builder.build_tree(spans)
    return tree_builder.to_json_string()


def process_spans_into_aggregated_traces(span_logs: DataFrame) -> DataFrame:
    """"""
    spark = init_spark()
    aggregated_traces = []
    traces = span_logs.groupBy(span_logs.trace_id).agg(collect_list('*'))
    for row in distinct_trace_id.collect():
        span_logs.where(span_logs.trace_id == row.trace_id)
        aggregated_traces.append(
            _construct_span_tree()
        )
    return spark.createDataFrame(data=aggregated_traces, schema=_get_aggregated_trace_log_spark_df_schema())
