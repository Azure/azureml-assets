# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Span Tree internal utilities and classes for building a Span Tree from preprocessed span logs."""


import bisect
from datetime import datetime
import json

from typing import Iterator, List
from pyspark.sql import Row


class SpanTreeNode:
    """Spantree node class."""

    def __init__(self, span_row: Row) -> None:
        """Represent a singular node in a span tree."""
        self._span_row = span_row
        self._children = []

    @property
    def span_id(self) -> str:
        """Get the span id."""
        return self._span_row.span_id

    @property
    def parent_id(self) -> str:
        """Get the span's parent id."""
        return self._span_row.parent_id

    @property
    def children(self) -> List["SpanTreeNode"]:
        """Get the span's children as list."""
        return self._children

    @children.setter
    def children(self, value: list) -> None:
        """Set the span's children."""
        self._children = value

    @property
    def span_row(self) -> Row:
        """Get the span's internal row."""
        return self._span_row

    @classmethod
    def create_node_from_json_str(cls, json_str: str) -> "SpanTreeNode":
        """Parse json string to get a single SpanTree node."""
        node_dict: dict = json.loads(json_str)

        node_dict['start_time'] = datetime.fromisoformat(node_dict['start_time'])
        node_dict['end_time'] = datetime.fromisoformat(node_dict['end_time'])
        children = node_dict.pop('children', [])
        new_row = Row(**node_dict)

        obj = cls.__new__(cls)
        super(SpanTreeNode, obj).__init__()
        obj._span_row = new_row
        obj._children = children
        return obj

    def insert_child(self, span: "SpanTreeNode") -> None:
        """Insert a child span in ascending time order due to __lt__()."""
        bisect.insort(self._children, span)

    def show(self, indent: int = 0) -> None:
        """Print the current span in a formatted syntax to stdout."""
        print(f"{' '*indent}[{self.span_row.span_id}({self.span_row.start_time}, {self.span_row.end_time})]")
        for c in self.children:
            c.show(indent + 4)

    def to_dict(self) -> dict:
        """Get dictionary representation of SpanTreeNode."""
        span_row_dict = self.span_row.asDict()
        span_row_dict['children'] = self.children
        return span_row_dict

    def __iter__(self) -> Iterator["SpanTreeNode"]:
        """Iterate over current span and child spans."""
        for child_span in self._children:
            for span in child_span:
                # print(f"iter func: child_span: {child_span}, span: {span}.")
                yield span
        yield self

    def __lt__(self, other: "SpanTreeNode") -> bool:
        """Compare by end_time in bisect.insort() for python3.8."""
        return self.span_row.end_time < other.span_row.end_time


class SpanTree:
    """Spantree class."""

    def __init__(self, spans: List[SpanTreeNode]) -> None:
        """Spantree constructor to build up tree from span list."""
        self.root_span = self._construct_span_tree(spans)
        self._load_json_tree_lazy = False

    @classmethod
    def create_tree_from_json_string(cls, json_string: str, load_tree_lazy: bool = False) -> "SpanTree":
        """Create SpanTree object from "root_span" json string."""
        obj = cls.__new__(cls)
        super(SpanTree, obj).__init__()
        obj._load_json_tree_lazy = load_tree_lazy
        # Default behavior is to load the whole tree from top level json string.
        # TODO: Stretch goal will be to load the tree lazily one level at a time to save computation/space.
        if not load_tree_lazy:
            obj.root_span = obj._from_json_str_repr(json_string)
        return obj

    def show(self) -> None:
        """Print to stdout a formatted representation of the Span Tree."""
        if self.root_span is None:
            return
        self.root_span.show()

    def to_json_str(self) -> str:
        """Get tree structure as json string."""
        if self.root_span is None:
            return None  # type: ignore
        return self._to_json_str_repr(self.root_span)

    def _construct_span_tree(self, spans: List[SpanTreeNode]) -> SpanTreeNode:
        """Build the span tree in ascending time order from list of all spans."""
        # construct a dict with span_id as key and span as value
        span_map = {span.span_id: span for span in spans}
        for span in span_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = span_map.get(parent_id)
                if parent_span is not None:
                    parent_span.insert_child(span)
        return root_span

    def __iter__(self) -> Iterator[SpanTreeNode]:
        """Iterate over the span tree in order."""
        if self.root_span is None:
            return
        for span in self.root_span.__iter__():
            yield span

    def _from_json_str_repr(self, json_string: str) -> SpanTreeNode:
        """Convert json string to SpanTree Node fully."""
        output_node = SpanTreeNode.create_node_from_json_str(json_string)
        child_subtree_nodes = []
        for child in output_node.children:
            new_node = self._from_json_str_repr(child)  # type: ignore
            child_subtree_nodes.append(new_node)
        output_node.children = child_subtree_nodes
        return output_node

    def _to_json_str_repr(self, curr_span: SpanTreeNode) -> str:
        """Recursively get tree structure JSON string."""
        output = curr_span.to_dict()
        start_time: datetime = output['start_time']  # type: ignore
        end_time: datetime = output['end_time']  # type: ignore
        output['start_time'] = start_time.isoformat()
        output['end_time'] = end_time.isoformat()

        child_subtree_strs = []
        for child in output['children']:
            subtree_str = self._to_json_str_repr(child)
            child_subtree_strs.append(subtree_str)
        output['children'] = child_subtree_strs
        return json.dumps(output)
