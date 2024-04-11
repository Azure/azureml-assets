# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Span Tree internal utilities and classes for building a Span Tree from preprocessed span logs."""


import bisect
from datetime import datetime
import json

from typing import Dict, Iterator, List, Optional
from pyspark.sql import Row
from copy import copy

from shared_utilities.constants import (
    APP_TRACES_EVENT_LOG_EMBEDDINGS_KEY,
    APP_TRACES_EVENT_LOG_INPUT_KEY,
    APP_TRACES_EVENT_LOG_OUTPUT_KEY,
    APP_TRACES_EVENT_LOG_RETRIEVAL_QUERY_KEY,
    APP_TRACES_EVENT_LOG_RETRIEVAL_DOCUMENT_KEY,
    EMBEDDING_SPAN_TYPE,
    RETRIEVAL_SPAN_TYPE
)
from shared_utilities.momo_exceptions import InvalidInputError


class SpanTreeNode:
    """Spantree node class."""

    def __init__(self, span_row: Row) -> None:
        """Represent a singular node in a span tree."""
        self._span_row: Row = span_row
        self._span_row_dict = {}
        if span_row is not None and span_row != Row():
            self._span_row_dict = span_row.asDict()
        self._children = []

    def get_node_attribute(self, attribute_key: str):
        """Get attribute from span row dictionary."""
        return self._span_row_dict.get(attribute_key, None)

    @property
    def span_id(self) -> str:
        """Get the span id."""
        return self.get_node_attribute("span_id")  # type: ignore

    @property
    def parent_id(self) -> str:
        """Get the span's parent id."""
        return self.get_node_attribute("parent_id")  # type: ignore

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
        return self.get_node_attribute("span_type")  # type: ignore

    @property
    def start_time(self) -> datetime:
        """Get the span's start_time."""
        return self.get_node_attribute("start_time")  # type: ignore

    @property
    def end_time(self) -> datetime:
        """Get the span's end_time."""
        return self.get_node_attribute("end_time")  # type: ignore

    @property
    def attributes(self) -> str:
        """Get the span's attributes."""
        return self.get_node_attribute("attributes")  # type: ignore

    @property
    def events(self) -> str:
        """Get the span's attributes."""
        return self.get_node_attribute("events")  # type: ignore

    @property
    def input(self) -> str:
        """Get the span's input from the events payload field as json string."""
        if self.events is None or len(self.events) == 0:
            return self.input_from_attributes()
        else:
            events_array: list = json.loads(self.events)
            if len(events_array) == 0:
                return self.input_from_attributes()
            input_event = list(filter(lambda event: event.get("name", None) == APP_TRACES_EVENT_LOG_INPUT_KEY,
                                    events_array))
            if input_event is not None and len(input_event) > 0:
                if input_event[0].get("attributes", None) is None:
                    return None
                return input_event[0].get("attributes", None).get("payload", None)
            else:
                return None

    def input_from_attributes(self) -> str:
        """Get input from attributes."""
        if self.attributes is None:

            return None  # type: ignore
        attribute_dict: dict = json.loads(self.attributes)
        if attribute_dict is None:
            return None
        return attribute_dict.get("inputs", None)

    @property
    def output(self) -> str:
        """Get the span's output from the events payload field as json string."""
        if self.events is None or len(self.events) == 0:
            return self.output_from_attribute()
        else:
            events_array: list = json.loads(self.events)
            if len(events_array) == 0:
                return self.output_from_attribute()
            output_event = list(filter(lambda event: event.get("name", None) == APP_TRACES_EVENT_LOG_OUTPUT_KEY,
                                       events_array))
            if output_event is not None and len(output_event) > 0:
                if output_event[0].get("attributes", None) is None:
                    return None
                return output_event[0].get("attributes", None).get("payload", None)
            else:
                return None

    def output_from_attribute(self) -> str:
        """Get output from attributes."""
        if self.attributes is None:
            return None  # type: ignore
        attribute_dict: dict = json.loads(self.attributes)
        if attribute_dict is None:
            return None
        return attribute_dict.get("output", None)

    @property
    def retrieval_query(self) -> str:
        """Get promptflow.retrieval.query for retrieval span. """
        if self.span_type != RETRIEVAL_SPAN_TYPE:
            return None
        if self.events is None or len(self.events) == 0:
            return self.retrieval_query_from_attributes()
        events_array: list = json.loads(self.events)
        if len(events_array) == 0:
            return self.retrieval_query_from_attributes()
        retrieval_event = list(filter(lambda event: event.get("name", None) ==
                                      APP_TRACES_EVENT_LOG_RETRIEVAL_QUERY_KEY,
                                      events_array))
        if retrieval_event and len(retrieval_event) > 0:
            if retrieval_event[0].get("attributes", None) is None:
                return None
            return retrieval_event[0].get("attributes", None).get("payload", None)
        else:
            return None
        
    @property
    def retrieval_document(self) -> str:
        """Get promptflow.retrieval.document for retrieval span. """
        if self.span_type != RETRIEVAL_SPAN_TYPE:
            return None
        if self.events is None or len(self.events) == 0:
            return self.retrieval_document_from_attributes()
        events_array: list = json.loads(self.events)
        if len(events_array) == 0:
            return self.retrieval_document_from_attributes()
        retrieval_event = list(filter(lambda event: event.get("name", None) ==
                                      APP_TRACES_EVENT_LOG_RETRIEVAL_DOCUMENT_KEY,
                                      events_array))
        if retrieval_event and len(retrieval_event) > 0:
            if retrieval_event[0].get("attributes", None) is None:
                return None
            return retrieval_event[0].get("attributes", None).get("payload", None)
        else:
            return None

    def retrieval_query_from_attributes(self) -> str:
        """Get retrieval query from attributes."""
        if self.attributes is None:
            return None
        attribute_dict: dict = json.loads(self.attributes)
        if attribute_dict is None:
            return None
        return attribute_dict.get("retrieval.query", None)

    def retrieval_document_from_attributes(self) -> str:
        """Get retrieval document from attributes."""
        if self.attributes is None:
            return None
        attribute_dict: dict = json.loads(self.attributes)
        if attribute_dict is None:
            return None
        return attribute_dict.get("retrieval.documents", None)

    @property
    def embeddings(self) -> str:
        """Get promptflow.embedding.embeddings for retrieval span. """
        if self.span_type != EMBEDDING_SPAN_TYPE:
            print("not span type ", self.span_type)
            return None
        if self.events is None or len(self.events) == 0:
            print("get embedding from attributes 1")
            return self.embeddings_from_attributes()
        events_array: list = json.loads(self.events)
        if len(events_array) == 0:
            print("get embedding from attributes 2")
            return self.embeddings_from_attributes()
        embeddings_event = list(filter(lambda event: event.get("name", None) ==
                                      APP_TRACES_EVENT_LOG_EMBEDDINGS_KEY,
                                      events_array))
        if embeddings_event and len(embeddings_event) > 0:
            if embeddings_event[0].get("attributes", None) is None:
                return None
            return embeddings_event[0].get("attributes", None).get("payload", None)
        else:
            return None

    def embeddings_from_attributes(self) -> str:
        """Get embeddings from attributes."""
        if self.attributes is None:
            return None
        attribute_dict: dict = json.loads(self.attributes)
        if attribute_dict is None:
            return None
        return attribute_dict.get("embedding.embeddings", None)

    @property
    def status(self) -> str:
        """Get the span's status."""
        return self.get_node_attribute("status")  # type: ignore

    @property
    def framework(self) -> str:
        """Get the span's framework."""
        return self.get_node_attribute("framework")  # type: ignore

    @property
    def name(self) -> str:
        """Get the span's name."""
        return self.get_node_attribute("name")  # type: ignore

    @property
    def trace_id(self) -> str:
        """Get the span's trace id."""
        return self.get_node_attribute("trace_id")  # type: ignore

    @classmethod
    def create_node_from_dict(cls, span_node_dict: dict) -> "SpanTreeNode":
        """Parse dict representation to create a single SpanTree node.

        NOTE: Do not call this function to deserialize/parse nodes individually. Instead use the
        `root_span` json string with `SpanTree.create_tree_from_json_string(root_span_string)`
        to deserialize all nodes in a tree.
        """
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
        obj._span_row_dict = span_node_dict
        obj._children = child_nodes
        return obj

    def insert_child(self, span: "SpanTreeNode") -> None:
        """Insert a child span in ascending time order due to __lt__()."""
        bisect.insort(self._children, span)

    def show(self, indent: int = 0) -> None:
        """Print the current span in a formatted syntax to stdout."""
        print(f"{' '*indent}[{self.span_id} ({self.start_time}, {self.end_time})]")
        for c in self.children:
            c.show(indent + 4)

    def to_dict(self, datetime_to_str: bool = True) -> dict:
        """Get dictionary representation of SpanTreeNode."""
        # map datetime object to iso-string and then turn children into list of dicts as well.
        output_dict = copy(self._span_row_dict)
        output_dict['children'] = self.children
        if datetime_to_str:
            start_time: datetime = output_dict['start_time']  # type: ignore
            end_time: datetime = output_dict['end_time']  # type: ignore
            output_dict['start_time'] = start_time.isoformat()
            output_dict['end_time'] = end_time.isoformat()

        child_subtree_dicts = []
        for child_node in output_dict['children'] or []:
            child_node: SpanTreeNode
            subtree_dict = child_node.to_dict()
            child_subtree_dicts.append(subtree_dict)

        output_dict['children'] = child_subtree_dicts
        return output_dict    

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
        self._span_node_map: Dict[str, SpanTreeNode] = {}
        self._root_span = self._construct_span_tree(spans)

    @property
    def root_span(self) -> SpanTreeNode:
        """Get the root span of the span tree."""
        return self._root_span

    @classmethod
    def create_tree_from_json_string(cls, json_string: str) -> "SpanTree":
        """Create SpanTree object from "root_span" json string."""
        obj = cls.__new__(cls)
        super(SpanTree, obj).__init__()
        # Default behavior is to load the whole tree from top level json string.
        root_span_dict = json.loads(json_string)
        if root_span_dict is None:
            obj._root_span = None
        else:
            obj._root_span = SpanTreeNode.create_node_from_dict(root_span_dict)
        obj._span_node_map = {span.span_id: span for span in obj}
        return obj

    def show(self) -> None:
        """Print to stdout a formatted representation of the Span Tree."""
        if self._root_span is None:
            print("The SpanTree is empty.")
            return
        print(f"SpanTree for trace id = {self._root_span.trace_id}:")
        self._root_span.show()

    def to_json_str(self) -> str:
        """Get tree structure as json string."""
        if self._root_span is None:
            return json.dumps(None)
        return json.dumps(self._root_span.to_dict())

    def get_span_tree_node_by_span_id(self, span_id: str) -> Optional[SpanTreeNode]:
        """Get a span tree node by span id. Return none if there is no matching span id."""
        if self._span_node_map is None:
            return None
        return self._span_node_map.get(span_id, None)

    def _construct_span_tree(self, spans: List[SpanTreeNode]) -> Optional[SpanTreeNode]:
        """Build the span tree in ascending time order from list of all spans."""
        root_span = None
        # construct a dict with span_id as key and span as value
        self._span_node_map = {span.span_id: span for span in spans}
        for span in self._span_node_map.values():
            parent_id = span.parent_id
            if parent_id is None:
                root_span = span
            else:
                parent_span = self.get_span_tree_node_by_span_id(parent_id)
                if parent_span is not None:
                    parent_span.insert_child(span)
        return root_span

    def __iter__(self) -> Iterator[SpanTreeNode]:
        """Iterate over the span tree in order."""
        if self._root_span is None:
            return
        for span in self._root_span.__iter__():
            yield span

    def __repr__(self) -> str:
        """Get representation of the SpanTree."""
        return f"SpanTree(trace id = {self._root_span.trace_id if self._root_span is not None else None}," + \
            f" spans = {self._span_node_map})"
