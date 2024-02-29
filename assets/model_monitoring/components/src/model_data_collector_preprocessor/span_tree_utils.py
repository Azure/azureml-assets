# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Span Tree internal utilities and classes for building a Span Tree from preprocessed span logs."""


import bisect
from datetime import datetime
import json

from typing import Iterator, List, Optional
from pyspark.sql import Row
from pyspark.sql.types import PySparkValueError

from shared_utilities.momo_exceptions import InvalidInputError


class SpanTreeNode:
    """Spantree node class."""

    def __init__(self, span_row: Row) -> None:
        """Represent a singular node in a span tree."""
        self._span_row = span_row
        self._children = []

    def _try_get_row_attribute(self, attribute_key: str):
        """Wrap span row retrieval to catch access errors."""
        try:
            return self._span_row[attribute_key]
        except PySparkValueError as ex:
            print(
                "Failed to retrieve row attribute with error: " +
                str(ex)
            )
            return None

    @property
    def span_id(self) -> str:
        """Get the span id."""
        return self._try_get_row_attribute("span_id")  # type: ignore

    @property
    def parent_id(self) -> str:
        """Get the span's parent id."""
        return self._try_get_row_attribute("parent_id")  # type: ignore

    @property
    def children(self) -> List["SpanTreeNode"]:
        """Get the span's children as list."""
        return self._children

    @children.setter
    def children(self, value: list) -> None:
        """Set the span's children."""
        self._children = value

    @property
    def span_type(self) -> str:
        """Get the span's type."""
        return self._try_get_row_attribute("span_type")  # type: ignore

    @property
    def start_time(self) -> datetime:
        """Get the span's start_time."""
        return self._try_get_row_attribute("start_time")  # type: ignore

    @property
    def end_time(self) -> datetime:
        """Get the span's end_time."""
        return self._try_get_row_attribute("end_time")  # type: ignore

    @property
    def attributes(self) -> str:
        """Get the span's attributes."""
        return self._try_get_row_attribute("attributes")  # type: ignore

    @property
    def input(self) -> str:
        """Get the span's input."""
        return self._try_get_row_attribute("input")  # type: ignore

    @property
    def output(self) -> str:
        """Get the span's output."""
        return self._try_get_row_attribute("output")  # type: ignore

    @property
    def status(self) -> str:
        """Get the span's status."""
        return self._try_get_row_attribute("status")  # type: ignore

    @property
    def framework(self) -> str:
        """Get the span's framework."""
        return self._try_get_row_attribute("framework")  # type: ignore

    @property
    def name(self) -> str:
        """Get the span's name."""
        return self._try_get_row_attribute("name")  # type: ignore

    @property
    def trace_id(self) -> str:
        """Get the span's trace id."""
        return self._try_get_row_attribute("trace_id")  # type: ignore

    @classmethod
    def create_node_from_dict(cls, span_node_dict: dict) -> "SpanTreeNode":
        """Parse dict representation to create a single SpanTree node."""
        if span_node_dict is None or span_node_dict == {}:
            raise InvalidInputError(
                "Can not create SpanTreeNode from empty root_span." +
                f" Input encountered: '{span_node_dict}'."
            )

        span_node_dict['start_time'] = datetime.fromisoformat(span_node_dict['start_time'])
        span_node_dict['end_time'] = datetime.fromisoformat(span_node_dict['end_time'])
        children_dicts = span_node_dict.pop('children', [])

        obj = cls.__new__(cls)
        super(SpanTreeNode, obj).__init__()

        child_nodes = []
        for child_dict in children_dicts:
            new_node = SpanTreeNode.create_node_from_dict(child_dict)
            child_nodes.append(new_node)

        obj._span_row = Row(**span_node_dict)
        obj._children = child_nodes
        return obj

    def insert_child(self, span: "SpanTreeNode") -> None:
        """Insert a child span in ascending time order due to __lt__()."""
        bisect.insort(self._children, span)

    def show(self, indent: int = 0) -> None:
        """Print the current span in a formatted syntax to stdout."""
        print(f"{' '*indent}[{self.span_id}({self.start_time}, {self.end_time})]")
        for c in self.children:
            c.show(indent + 4)

    def to_dict(self, datetime_to_str: bool = True) -> dict:
        """Get dictionary representation of SpanTreeNode."""
        # map datetime object to iso-string and then turn children into list of dicts as well.
        span_row_dict = self._span_row.asDict()
        span_row_dict['children'] = self.children
        if datetime_to_str:
            start_time: datetime = span_row_dict['start_time']  # type: ignore
            end_time: datetime = span_row_dict['end_time']  # type: ignore
            span_row_dict['start_time'] = start_time.isoformat()
            span_row_dict['end_time'] = end_time.isoformat()

        child_subtree_dicts = []
        for child_node in span_row_dict['children'] or []:
            child_node: SpanTreeNode
            subtree_dict = child_node.to_dict()
            child_subtree_dicts.append(subtree_dict)

        span_row_dict['children'] = child_subtree_dicts
        return span_row_dict

    def __iter__(self) -> Iterator["SpanTreeNode"]:
        """Iterate over current span and child spans."""
        for child_span in self._children or []:
            for span in child_span:
                yield span
        yield self

    def __lt__(self, other: "SpanTreeNode") -> bool:
        """Compare by end_time in bisect.insort() for python3.8."""
        return self.end_time < other.end_time

    def __repr__(self) -> str:
        """Get representation of SpanTreeNode."""
        return f"SpanTreeNode(span_id: {self.span_id}, trace_id: {self.trace_id})"


class SpanTree:
    """Spantree class."""

    def __init__(self, spans: List[SpanTreeNode]) -> None:
        """Spantree constructor to build up tree from span list."""
        self._span_node_map: dict = {}
        self.root_span = self._construct_span_tree(spans)

    @classmethod
    def create_tree_from_json_string(cls, json_string: str) -> "SpanTree":
        """Create SpanTree object from "root_span" json string."""
        obj = cls.__new__(cls)
        super(SpanTree, obj).__init__()
        # Default behavior is to load the whole tree from top level json string.
        root_span_dict = json.loads(json_string)
        if root_span_dict is None:
            obj.root_span = None
        else:
            obj.root_span = SpanTreeNode.create_node_from_dict(root_span_dict)
        return obj

    def show(self) -> None:
        """Print to stdout a formatted representation of the Span Tree."""
        if self.root_span is None:
            return
        self.root_span.show()

    def to_json_str(self) -> str:
        """Get tree structure as json string."""
        if self.root_span is None:
            return json.dumps(None)
        return json.dumps(self.root_span.to_dict())

    def get_span_tree_node_by_span_id(self, span_id: str) -> Optional[SpanTreeNode]:
        """Get a span tree node by span id. Return none if there is no matching span id."""
        if self._span_node_map is None:
            return None
        return self._span_node_map.get(span_id, None)

    def _construct_span_tree(self, spans: List[SpanTreeNode]) -> SpanTreeNode:
        """Build the span tree in ascending time order from list of all spans."""
        # construct a dict with span_id as key and span as value
        root_span = None
        self._span_node_map = {span.span_id: span for span in spans}
        for span in self._span_node_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = self.get_span_tree_node_by_span_id(parent_id)
                if parent_span is not None:
                    parent_span.insert_child(span)
        # todo: handle logic if root_span is not found or if we have multiple root_spans
        return root_span

    def __iter__(self) -> Iterator[SpanTreeNode]:
        """Iterate over the span tree in order."""
        if self.root_span is None:
            return
        for span in self.root_span.__iter__():
            yield span
