# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for SpanTreeNode(Row Tree Utilities."""


from pyspark.sql import Row
from src.model_data_collector_preprocessor.span_tree.span_tree_utils import (
    SpanTree,
    SpanTreeNode,
    _get_span_tree_node_spark_df_schema
)
from tests.e2e.utils.io_utils import create_pyspark_dataframe
import pytest
from datetime import datetime


@pytest.mark.unit
class TestSpanTreeUtilities:
    """Test class for span Row Tree Utilities."""

    ################### SpanTree class tests: ###################

    def test_span_tree_construct(self):
        """Test basic scenario to construct span tree with ascending time order."""
        # The data end, start times in easy to read format:
        # s0 = 0, 100
        # s00 = 5, 30
        # s01 = 35, 60
        # s02 = 65, 90
        # s010 = 40, 50
        s0 = SpanTreeNode(Row(span_id="0", parent_id=None, start_time="2024-02-12T00:00:01", end_time="2024-02-12T01:40:00"))
        s00 = SpanTreeNode(Row(span_id="00", parent_id="0", start_time="2024-02-12T00:05:00", end_time="2024-02-12T00:30:00"))
        s01 = SpanTreeNode(Row(span_id="01", parent_id="0", start_time="2024-02-12T00:35:00", end_time="2024-02-12T01:00:00"))
        s02 = SpanTreeNode(Row(span_id="02", parent_id="0", start_time="2024-02-12T01:05:00", end_time="2024-02-12T01:30:00"))
        s010 = SpanTreeNode(Row(span_id="010", parent_id="01", start_time="2024-02-12T00:40:00", end_time="2024-02-12T00:50:00"))
        spans = [s0, s02, s010, s00, s01]

        tree = SpanTree(spans)
        curr_end_time = datetime.fromisoformat("2024-02-12T00:00:00")
        for span in tree:
            next_end_time = datetime.fromisoformat(span.span_row["end_time"])
            assert curr_end_time < next_end_time
            curr_end_time = next_end_time

    ################### End ###################

    ################### SpanTreeNode class tests: ###################

    def test_span_tree_node_children(self):
        """Test scenarios for inserting child span tree nodes."""
        node10 = SpanTreeNode(Row(end_time='2024-02-12T00:10:00'))
        node20 = SpanTreeNode(Row(end_time='2024-02-12T00:20:00'))
        node30 = SpanTreeNode(Row(end_time='2024-02-12T00:30:00'))
        node40 = SpanTreeNode(Row(end_time='2024-02-12T00:40:00'))

        parent_node = SpanTreeNode(Row())
        assert parent_node.children == []

        parent_node.insert_child(node30)
        assert len(parent_node.children) == 1
        assert node30.span_row['end_time'] == parent_node.children[0].span_row['end_time']

        parent_node.insert_child(node10)
        assert len(parent_node.children) == 2
        assert node10.span_row['end_time'] == parent_node.children[0].span_row['end_time']

        parent_node.insert_child(node20)
        assert len(parent_node.children) == 3
        assert node20.span_row['end_time'] == parent_node.children[1].span_row['end_time']

        parent_node.insert_child(node40)
        assert len(parent_node.children) == 4
        assert node40.span_row['end_time'] == parent_node.children[-1].span_row['end_time']

    @pytest.mark.parametrize(
            "expected_children_arrays",
            [
                ([]),
                (["test1", 1, 2, "test3"]),
                ([SpanTreeNode(Row()), SpanTreeNode(Row())]),
                (None)
            ]

    )
    def test_span_tree_node_children_property(self, expected_children_arrays: list):
        """Test scenarios for setting and getting span tree node children."""
        node = SpanTreeNode(Row())
        node.children = expected_children_arrays
        assert expected_children_arrays == node.children

    @pytest.mark.parametrize(
            "expected_row",
            [
                (Row(span_id="0", parent_id=None, start_time="2024-02-12T00:00:01", end_time="2024-02-12T01:40:00")),
                (Row(span_id="00", parent_id="0", start_time="2024-02-12T00:05:00", end_time="2024-02-12T00:30:00")),
                (Row(span_id=None, parent_id=None, start_time=None, end_time=None))
            ]
    )
    def test_span_tree_node_other_properties(self, expected_row: Row):
        """Test scenarios for getting the other tree node properties"""
        node = SpanTreeNode(expected_row)
        assert expected_row.span_id == node.span_id
        assert expected_row.parent_id == node.parent_id
        assert expected_row == node.span_row

    def test_span_tree_node_create_from_json(self):
        """Test scenario for creating new node from json string."""
        json_string = ""
        try:
            node = SpanTreeNode.create_node_from_json_str(json_string)
            assert False
        except:
            assert True

        json_string = '{"parent_id": null,"span_id": "0x7fd179134fb9e709","span_type": "SpanKind.INTERNAL",' + \
            '"start_time": "2024-02-05T15:00:13.789782Z","end_time": "2024-02-05T15:00:18.791564Z",' + \
            '"children": ["fakejsonchild", "fakejsonchild2"]}'
        expected_row = Row(
            parent_id=None, span_id="0x7fd179134fb9e709", span_type="SpanKind.INTERNAL",
            start_time="2024-02-05T15:00:13.789782Z", end_time="2024-02-05T15:00:18.791564Z",
        )
        expected_children = ["fakejsonchild", "fakejsonchild2"]

        node = SpanTreeNode.create_node_from_json_str(json_string)
        assert node.span_id == "0x7fd179134fb9e709"
        assert node.parent_id == None
        assert node.span_row == expected_row
        assert node.children == expected_children

    @pytest.mark.parametrize(
            "expected_row, expected_children",
            [
                (
                    Row(
                        span_id="0",
                        parent_id=None,
                        span_type="SpanKind.INTERNAL",
                        start_time="2024-02-12T00:00:01",
                        end_time="2024-02-12T01:40:00",
                    ),
                    ["fake child"]
                ),
                (
                    Row(
                        span_id="00",
                        parent_id="0",
                        span_type="SpanKind.TOOL",
                        start_time="2024-02-12T00:05:00",
                        end_time="2024-02-12T00:30:00",
                    ),
                    []
                ),
                (
                    Row(
                        span_id=None,
                        parent_id=None,
                        span_type=None,
                        start_time=None,
                        end_time=None,
                    ),
                    None
                )
            ]
    )
    def test_span_tree_node_to_dict(self, expected_row: Row, expected_children):
        """Test scenario for to_dict() of SpanTreeNode."""
        dict_schema_keynames = _get_span_tree_node_spark_df_schema().fieldNames()
        node = SpanTreeNode(expected_row)
        node.children = expected_children
        actual_dict = node.to_dict()
        for keyname in dict_schema_keynames:
            assert keyname in actual_dict
            if keyname == "children":
                assert actual_dict[keyname] == expected_children
            else:
                assert actual_dict[keyname] == node.span_row[keyname]

    ################### Span Tree Node class tests: ###################
