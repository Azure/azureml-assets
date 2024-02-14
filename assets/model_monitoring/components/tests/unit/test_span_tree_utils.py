# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for SpanTreeNode(Row Tree Utilities."""


from pyspark.sql import Row
from assets.model_monitoring.components.src.model_data_collector_preprocessor.span_tree_utils import (
    SpanTree,
    SpanTreeNode,
    _get_span_tree_node_spark_df_schema
)

import pytest
from datetime import datetime


@pytest.mark.unit
class TestSpanTreeUtilities:
    """Test class for span Row Tree Utilities."""

    def test_span_tree_construct(self):
        """Test basic scenario to construct span tree with ascending time order."""
        # The data end, start times in easy to read format:
        # s0 = 0, 100
        # s00 = 5, 30
        # s01 = 35, 60
        # s02 = 65, 90
        # s010 = 40, 50
        s0 = SpanTreeNode(
            Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                end_time=datetime(2024, 2, 12, 1, 40, 0))
        )
        s00 = SpanTreeNode(
            Row(span_id="00", parent_id="0", start_time=datetime(2024, 2, 12, 0, 5, 0),
                end_time=datetime(2024, 2, 12, 0, 30, 0))
        )
        s01 = SpanTreeNode(
            Row(span_id="01", parent_id="0", start_time=datetime(2024, 2, 12, 0, 35, 0),
                end_time=datetime(2024, 2, 12, 1, 0, 0))
        )
        s02 = SpanTreeNode(
            Row(span_id="02", parent_id="0", start_time=datetime(2024, 2, 12, 1, 5, 0),
                end_time=datetime(2024, 2, 12, 1, 30, 0))
        )
        s010 = SpanTreeNode(
            Row(span_id="010", parent_id="01", start_time=datetime(2024, 2, 12, 0, 40, 0),
                end_time=datetime(2024, 2, 12, 0, 50, 0))
        )
        spans = [s0, s02, s010, s00, s01]

        tree = SpanTree(spans)
        curr_end_time = datetime.fromisoformat("2024-02-12T00:00:00")
        for span in tree:
            next_end_time = span.span_row["end_time"]
            assert curr_end_time < next_end_time
            curr_end_time = next_end_time

    def test_span_tree_from_json_string(self):
        """Test scenario to construct span tree from json string."""
        json_string = ""
        try:
            SpanTree.create_tree_from_json_string(json_string)
            assert False
        except Exception:
            assert True

        # The tree is structured like this:
        # 1
        # |-> 2 -> 3
        # |-> 4
        json_string = '{"parent_id": null, "span_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:' + \
            '01:00", "end_time": "2024-02-05T00:08:00", "children": ["{\\"parent_id\\": \\"1\\", \\"span_id\\"' + \
            ': \\"2\\", \\"span_type\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:02:00\\", \\"end_time\\"' + \
            ': \\"2024-02-05T00:05:00\\", \\"children\\": [\\"{\\\\\\"parent_id\\\\\\": \\\\\\"2\\\\\\"' + \
            ', \\\\\\"span_id\\\\\\": \\\\\\"3\\\\\\", \\\\\\"span_type\\\\\\": \\\\\\"llm\\\\\\", \\\\\\"' + \
            'start_time\\\\\\": \\\\\\"2024-02-05T00:03:00\\\\\\", \\\\\\"end_time\\\\\\": \\\\\\"2024-02-05' + \
            'T00:04:00\\\\\\", \\\\\\"children\\\\\\": []}\\"]}", "{\\"parent_id\\": \\"1\\", \\"span_id\\":' + \
            ' \\"4\\", \\"span_type\\": \\"llm\\", \\"start_time\\": \\"2024-02-05T00:06:00\\", \\"end_time\\"' + \
            ': \\"2024-02-05T00:07:00\\", \\"children\\": []}"]}'
        tree = SpanTree.create_tree_from_json_string(json_string)

        assert "1" == tree.root_span.span_id
        assert tree.root_span.parent_id is None
        assert datetime(2024, 2, 5, 0, 8, 0) == tree.root_span.span_row.end_time
        assert datetime(2024, 2, 5, 0, 1, 0) == tree.root_span.span_row.start_time
        assert 2 == len(tree.root_span.children)

        first_child = tree.root_span.children[0]
        assert "2" == first_child.span_id
        assert "1" == first_child.parent_id
        assert datetime(2024, 2, 5, 0, 5, 0) == first_child.span_row.end_time
        assert datetime(2024, 2, 5, 0, 2, 0) == first_child.span_row.start_time
        assert 1 == len(first_child.children)

        second_child = first_child.children[0]
        assert "3" == second_child.span_id
        assert "2" == second_child.parent_id
        assert datetime(2024, 2, 5, 0, 4, 0) == second_child.span_row.end_time
        assert datetime(2024, 2, 5, 0, 3, 0) == second_child.span_row.start_time
        assert 0 == len(second_child.children)

        third_child = tree.root_span.children[1]
        assert "4" == third_child.span_id
        assert "1" == third_child.parent_id
        assert datetime(2024, 2, 5, 0, 7, 0) == third_child.span_row.end_time
        assert datetime(2024, 2, 5, 0, 6, 0) == third_child.span_row.start_time
        assert 0 == len(third_child.children)

    def test_span_tree_node_children(self):
        """Test scenarios for inserting child span tree nodes."""
        node10 = SpanTreeNode(Row(end_time=datetime(2024, 2, 12, 0, 10)))
        node20 = SpanTreeNode(Row(end_time=datetime(2024, 2, 12, 0, 20)))
        node30 = SpanTreeNode(Row(end_time=datetime(2024, 2, 12, 0, 30)))
        node40 = SpanTreeNode(Row(end_time=datetime(2024, 2, 12, 0, 40)))

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
                (Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                     end_time=datetime(2024, 2, 12, 1, 40, 0))),
                (Row(span_id="00", parent_id="0", start_time=datetime(2024, 2, 12, 0, 5, 0),
                     end_time=datetime(2024, 2, 12, 0, 30, 0))),
                (Row(span_id=None, parent_id=None, start_time=None, end_time=None))
            ]
    )
    def test_span_tree_node_other_properties(self, expected_row: Row):
        """Test scenarios for getting the other tree node properties."""
        node = SpanTreeNode(expected_row)
        assert expected_row.span_id == node.span_id
        assert expected_row.parent_id == node.parent_id
        assert expected_row == node.span_row

    def test_span_tree_node_create_from_json(self):
        """Test scenario for creating new node from json string."""
        json_string = ""
        try:
            SpanTreeNode.create_node_from_json_str(json_string)
            assert False
        except Exception:
            assert True

        json_string = '{"parent_id":null,"span_id":"0x7fd179134fb9e709","span_type":"SpanKind.INTERNAL",' + \
            '"start_time":"2024-02-05T15:00:13.789782","end_time":"2024-02-05T15:00:18.791564",' + \
            '"children":["fakejsonchild", "fakejsonchild2"]}'
        expected_row = Row(
            parent_id=None, span_id="0x7fd179134fb9e709", span_type="SpanKind.INTERNAL",
            start_time=datetime(2024, 2, 5, 15, 0, 13, 789782), end_time=datetime(2024, 2, 5, 15, 0, 18, 791564),
        )
        expected_children = ["fakejsonchild", "fakejsonchild2"]

        node = SpanTreeNode.create_node_from_json_str(json_string)
        assert node.span_id == "0x7fd179134fb9e709"
        assert node.parent_id is None
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
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                    ),
                    ["fake child"]
                ),
                (
                    Row(
                        span_id="00",
                        parent_id="0",
                        span_type="SpanKind.TOOL",
                        start_time=datetime(2024, 2, 12, 0, 5, 0),
                        end_time=datetime(2024, 2, 12, 0, 30, 0),
                    ),
                    []
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
            elif keyname == "start_time" or keyname == "end_time":
                assert datetime.fromisoformat(actual_dict[keyname]) == node.span_row[keyname]
            else:
                assert actual_dict[keyname] == node.span_row[keyname]
