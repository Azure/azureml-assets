# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for SpanTreeNode(Row Tree Utilities."""

import json
import pytest
from datetime import datetime
from pyspark.sql import Row
from src.shared_utilities.span_tree_utils import (
    SpanTree,
    SpanTreeNode,
)


@pytest.mark.unit
class TestSpanTreeUtilities:
    """Test class for span Row Tree Utilities."""

    def test_span_tree_construct_and_to_json(self):
        """Test basic scenario to construct span tree with ascending time order and convert to json."""
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

        # Test the __iter__ functionality also
        curr_end_time = datetime.fromisoformat("2024-02-12T00:00:00")
        for span in tree:
            next_end_time = span.end_time
            assert curr_end_time < next_end_time
            curr_end_time = next_end_time

        json_str = tree.to_json_str()
        json_tree = SpanTree.create_tree_from_json_string(json_str)
        expected_span_id_order = ["00", "010", "01", "02", "0"]

        for actual_span, expected_span_id in zip(json_tree, expected_span_id_order):
            assert expected_span_id == actual_span.span_id

    def test_span_tree_construct_no_root_span(self):
        """Test various scenarios where we span log data with no root_span for various reason."""
        s0 = SpanTreeNode(
            Row(trace_id="01", span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 9, 0, 0),
                end_time=datetime(2024, 2, 12, 10, 5, 0)))
        s1 = SpanTreeNode(
            Row(trace_id="01", span_id="1", parent_id="0", start_time=datetime(2024, 2, 12, 9, 5, 0),
                end_time=datetime(2024, 2, 12, 9, 40, 0))
        )
        s2 = SpanTreeNode(
            Row(trace_id="01", span_id="2", parent_id="1", start_time=datetime(2024, 2, 12, 9, 15, 0),
                end_time=datetime(2024, 2, 12, 9, 20, 0))
        )
        s3 = SpanTreeNode(
            Row(trace_id="01", span_id="3", parent_id="1", start_time=datetime(2024, 2, 12, 9, 25, 0),
                end_time=datetime(2024, 2, 12, 9, 30, 0))
        )
        s4 = SpanTreeNode(
            Row(trace_id="01", span_id="4", parent_id="0", start_time=datetime(2024, 2, 12, 9, 45, 0),
                end_time=datetime(2024, 2, 12, 9, 50, 0))
        )
        s5 = SpanTreeNode(
            Row(trace_id="01", span_id="5", parent_id="00", start_time=datetime(2024, 2, 12, 10, 10, 0),
                end_time=datetime(2024, 2, 12, 10, 15, 0))
        )
        empty_spans = []
        empty_tree = SpanTree(empty_spans)
        empty_tree.show()

        assert empty_tree.root_span is None
        assert empty_tree._span_node_map == {}

        # scenario where single span is given and parent_id is None.
        # root span should be 0.
        spans_0 = [s0]
        tree_0 = SpanTree(spans_0)

        print("tree 0:")
        tree_0.show()
        print()

        assert tree_0.root_span is not None
        assert tree_0.root_span == s0
        assert tree_0.root_span.children == []

        # scneario 1 where single span is give but parent_id is pointing to an unknown node.
        #    X
        # -- | -----
        #    | -> 1

        spans_1 = [s1]
        tree_1 = SpanTree(spans_1)

        print("tree 1:")
        tree_1.show()
        print()

        assert tree_1.root_span is None
        assert tree_1._span_node_map == {span.span_id: span for span in spans_1}
        # TODO: Uncomment if we ever want to handle scenarios like this differently.
        # root span should be 1.
        # assert tree_1.root_span is not None
        # assert tree_1.root_span == s1
        # assert tree_1.root_span.children == []

        # scenario 2 where root span "0" is outside our data_window. Visually denoted by the dashes
        #    0
        # -- | -----
        #    | -> 1 -> 2
        #         | -> 3

        spans_2 = [s1, s2, s3]
        tree_2 = SpanTree(spans_2)
        print("tree 2:")
        tree_2.show()
        print()

        assert tree_2.root_span is None
        assert tree_2._span_node_map == {span.span_id: span for span in spans_2}
        # TODO: Uncomment if we ever want to handle scenarios like this differently.
        # root span should be 1.
        # assert tree_2.root_span is not None
        # assert tree_2.root_span == s1
        # assert tree_2.root_span.children[0] == s2
        # assert tree_2.root_span.children[1] == s3

        # scenario 3 where root span "0" is outside our data_window but have multiple spans pointing to it.
        #    0
        # -- | -----
        #    | -> 1 -> 2
        #    |    | -> 3
        #    | -> 4

        # root span is undetermined.
        spans_3 = [s1, s2, s3, s4]
        tree_3 = SpanTree(spans_3)
        print("tree 3:")
        tree_3.show()
        print()

        assert tree_3.root_span is None
        assert tree_3._span_node_map == {span.span_id: span for span in spans_3}
        # TODO: uncomment when we know what to look for in this case.
        # assert tree_3.root_span is not None
        # assert tree_3.root_span == s1
        # assert tree_3.root_span.children[0] == s2
        # assert tree_3.root_span.children[1] == s4

        # scenario 4 where spans in our data point to multiple outside parent spans.
        # 00  0
        # -|--| -----
        #  |  | -> 1 -> 2
        #  |  |    | -> 3
        #  |  | -> 4
        #  |-> 5

        spans_4 = [s1, s2, s3, s4, s5]
        tree_4 = SpanTree(spans_4)
        print("tree 4:")
        tree_4.show()
        print()

        assert tree_4.root_span is None
        assert tree_4._span_node_map == {span.span_id: span for span in spans_4}
        # TODO: uncomment when we know what to look for in this case.
        # assert tree_4.root_span is not None
        # assert tree_4.root_span == s1
        # assert tree_4.root_span.children[0] == s2
        # assert tree_4.root_span.children[1] == s4

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
            '01:00", "end_time": "2024-02-05T00:08:00", "children": [{"parent_id": "1", "span_id"' + \
            ': "2", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "end_time"' + \
            ': "2024-02-05T00:05:00", "children": [{"parent_id": "2", "span_id": "3", "span_type": "llm", "' + \
            'start_time": "2024-02-05T00:03:00", "end_time": "2024-02-05' + \
            'T00:04:00", "children": []}]}, {"parent_id": "1", "span_id": "4", "span_type": "llm", ' + \
            '"start_time": "2024-02-05T00:06:00", "end_time": "2024-02-05T00:07:00", "children": []}]}'
        tree = SpanTree.create_tree_from_json_string(json_string)

        assert "1" == tree.root_span.span_id
        assert tree.root_span.parent_id is None
        assert datetime(2024, 2, 5, 0, 8, 0) == tree.root_span.end_time
        assert datetime(2024, 2, 5, 0, 1, 0) == tree.root_span.start_time
        assert 2 == len(tree.root_span.children)

        first_child = tree.root_span.children[0]
        assert "2" == first_child.span_id
        assert "1" == first_child.parent_id
        assert datetime(2024, 2, 5, 0, 5, 0) == first_child.end_time
        assert datetime(2024, 2, 5, 0, 2, 0) == first_child.start_time
        assert 1 == len(first_child.children)

        second_child = first_child.children[0]
        assert "3" == second_child.span_id
        assert "2" == second_child.parent_id
        assert datetime(2024, 2, 5, 0, 4, 0) == second_child.end_time
        assert datetime(2024, 2, 5, 0, 3, 0) == second_child.start_time
        assert 0 == len(second_child.children)

        third_child = tree.root_span.children[1]
        assert "4" == third_child.span_id
        assert "1" == third_child.parent_id
        assert datetime(2024, 2, 5, 0, 7, 0) == third_child.end_time
        assert datetime(2024, 2, 5, 0, 6, 0) == third_child.start_time
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
        assert node30.end_time == parent_node.children[0].end_time

        parent_node.insert_child(node10)
        assert len(parent_node.children) == 2
        assert node10.end_time == parent_node.children[0].end_time

        parent_node.insert_child(node20)
        assert len(parent_node.children) == 3
        assert node20.end_time == parent_node.children[1].end_time

        parent_node.insert_child(node40)
        assert len(parent_node.children) == 4
        assert node40.end_time == parent_node.children[-1].end_time

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
                     end_time=datetime(2024, 2, 12, 1, 40, 0), trace_id="1", status="OK",
                     attributes="{\"inputs\": \"null\"}", span_type="SpanKind.INTERNAL",
                     input="{\"context\":\"...\", \"ground_truth\":\"...\"}", name="LLM", framework="LLM")),
                (Row(span_id="00", parent_id="0", start_time=datetime(2024, 2, 12, 0, 5, 0),
                     end_time=datetime(2024, 2, 12, 0, 30, 0), trace_id="1", status="",
                     attributes="{}", span_type="SpanKind.INTERNAL", name="", framework="")),
                (Row(span_id=None, parent_id=None, start_time=None, end_time=None,
                     trace_id=None, status=None, attributes=None, span_type=None, name=None, framework=None)),
                (Row(span_id=None, parent_id=None, start_time=None, end_time=None,
                     trace_id=None, status=None, attributes="null", span_type=None, name=None, framework=None)),
            ]
    )
    def test_span_tree_node_other_properties(self, expected_row: Row):
        """Test scenarios for getting the other tree node properties."""
        node = SpanTreeNode(expected_row)
        assert expected_row.span_id == node.span_id
        assert expected_row.parent_id == node.parent_id
        assert expected_row.trace_id == node.trace_id
        assert expected_row.status == node.status
        assert expected_row.attributes == node.attributes
        assert expected_row.start_time == node.start_time
        assert expected_row.end_time == node.end_time
        assert expected_row.span_type == node.span_type

        if expected_row.attributes is not None:
            attributes_dict = json.loads(expected_row.attributes)
            attributes_dict = {} if attributes_dict is None else attributes_dict
            assert attributes_dict.get('inputs', None) == node.input
            assert attributes_dict.get('output', None) == node.output
        else:
            assert node.input is None
            assert node.output is None
        assert expected_row.name == node.name
        assert expected_row.framework == node.framework
        assert expected_row == node._span_row

    def test_span_tree_node_create_from_dict(self):
        """Test scenario for creating new node from json string."""
        json_dict = {}
        try:
            SpanTreeNode.create_node_from_dict(json_dict)
            assert False
        except Exception:
            assert True

        json_dict = None
        try:
            SpanTreeNode.create_node_from_dict(json_dict)
            assert False
        except Exception:
            assert True

        json_dict = {
            "parent_id": None, "span_id": "0x7fd179134fb9e709", "span_type": "SpanKind.INTERNAL",
            "start_time": "2024-02-05T15:00:13.789782", "end_time": "2024-02-05T15:00:18.791564",
            "children": [
                {"name": "fakejsonchild", "start_time": "2024-02-05T15:00:14", "end_time": "2024-02-05T15:00:15"},
                {"name": "fakejsonchild2", "start_time": "2024-02-05T15:00:16", "end_time": "2024-02-05T15:00:17"}
            ]
        }
        expected_row = Row(
            parent_id=None, span_id="0x7fd179134fb9e709", span_type="SpanKind.INTERNAL",
            start_time=datetime(2024, 2, 5, 15, 0, 13, 789782), end_time=datetime(2024, 2, 5, 15, 0, 18, 791564),
        )
        expected_children = [
            SpanTreeNode(Row(
                **{"name": "fakejsonchild", "start_time": datetime(2024, 2, 5, 15, 0, 14),
                   "end_time": datetime(2024, 2, 5, 15, 0, 15)})),
            SpanTreeNode(Row(
                **{"name": "fakejsonchild2", "start_time": datetime(2024, 2, 5, 15, 0, 16),
                   "end_time": datetime(2024, 2, 5, 15, 0, 17)})),
        ]

        node = SpanTreeNode.create_node_from_dict(json_dict)
        assert node.span_id == "0x7fd179134fb9e709"
        assert node.parent_id is None
        assert node._span_row == expected_row
        assert node.children[0]._span_row == expected_children[0]._span_row
        assert node.children[1]._span_row == expected_children[1]._span_row
        assert node._span_row_dict == json_dict

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
                    [SpanTreeNode(Row(**{
                        "span_id": "1",
                        "parent_id": "0",
                        "start_time": datetime(2024, 2, 12, 0, 0, 2),
                        "end_time": datetime(2024, 2, 12, 0, 0, 5),
                    }))]
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
        node = SpanTreeNode(expected_row)
        node.children = expected_children
        for datetime_to_str in [True, False]:
            actual_dict = node.to_dict(datetime_to_str)
            for keyname in expected_row.asDict().keys():
                assert keyname in actual_dict
                if keyname == "children":
                    assert actual_dict[keyname] == expected_children
                elif keyname == "start_time" or keyname == "end_time":
                    expected_datetime: datetime = node._span_row[keyname]
                    if datetime_to_str:
                        assert actual_dict[keyname] == expected_datetime.isoformat()
                    else:
                        assert actual_dict[keyname] == expected_datetime
                else:
                    assert actual_dict[keyname] == node._span_row[keyname]

    def test_span_tree_get_span_by_span_id(self):
        """Test scenarios for get_span_by_id()."""
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
        assert tree.get_span_tree_node_by_span_id("") is None
        assert tree.get_span_tree_node_by_span_id(None) is None  # type: ignore
        assert tree.get_span_tree_node_by_span_id("0") == s0

    @pytest.mark.parametrize(
            "row,inputs,output",
            [
                (
                    Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0), trace_id="1", status="OK",
                        attributes=None,
                        events=None),
                    None,
                    None),
                (
                    Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0), trace_id="1", status="OK",
                        attributes=None,
                        events="[{\"name\":\"promptflow.function.inputs\",\"timestamp\":"
                        "\"2024-04-09T01:42:23.587010Z\","
                        "\"attributes\":{\"payload\":\"{\\n  \\\"model\\\": "
                        "\\\"gpt-4-0405\\\",\\n  \\\"messages\\\":"
                        " [\\n    {\\n      \\\"role\\\": \\\"system\\\",\\n      "
                        "\\\"content\\\": \\\"You are "
                        "a helpful assistant.\\\"\\n    },\\n    {\\n      "
                        "\\\"role\\\": \\\"user\\\",\\n      "
                        "\\\"content\\\": \\\"hello\\\"\\n    },\\n    {\\n      "
                        "\\\"role\\\": \\\"assistant\\\",\\n"
                        "      \\\"content\\\": \\\"Hello! How can I assist you today?\\\""
                        "\\n    },\\n    {\\n      "
                        "\\\"role\\\": \\\"user\\\",\\n      \\\"content\\\": "
                        "\\\"what can you do?\\\"\\n    }\\n  ],\\n"
                        "  \\\"temperature\\\": 0.7,\\n  \\\"top_p\\\": 1.0,\\n  "
                        "\\\"n\\\": 1,\\n  \\\"stream\\\": false,\\n"
                        "  \\\"presence_penalty\\\": 0.0,\\n  "
                        "\\\"frequency_penalty\\\": 0.0,\\n  \\\"user\\\": \\\"\\\",\\n"
                        "  \\\"max_tokens\\\": 256\\n}\"}},{\"name\":"
                        "\"promptflow.function.output\",\"timestamp\""
                        ":\"2024-04-09T01:42:50.757239Z\",\"attributes\":{"
                        "\"payload\":\"{\\n  \\\"id\\\": "
                        "\\\"chatcmpl-9Bun9MD3qDvYLvVeGenY4uF6sLyS1\\\",\\n  "
                        "\\\"choices\\\": [\\n    {\\n"
                        "      \\\"finish_reason\\\": \\\"length\\\",\\n      "
                        "\\\"index\\\": 0,\\n      "
                        "\\\"logprobs\\\": null,\\n      \\\"message\\\": {\\n        "
                        "\\\"content\\\": \\\"I can "
                        "help with a wide range of questions and tasks! "
                        "Here are some examples:\\\\n\\\\n1. "
                        "**General Knowledge and Facts**: I can provide "
                        "information on a vast array of topics "
                        "such as history, science, technology, and more.\\\\n2. "
                        "**Educational Assistance**: "
                        "I can help with explanations, summaries, and information on various academic subjects."
                        "\\\\n3. **Daily Tasks**: I can assist with setting reminders, providing weather updates, "
                        "or offering advice on daily routines.\\\\n4. **Technical Support**: I can offer guidance on"
                        " using software, troubleshooting basic tech issues, or understanding tech concepts.\\\\n5. "
                        "**Entertainment and Recommendations**: "
                        "I can suggest books, movies, music, or games based on "
                        "your preferences.\\\\n6. **Travel and Geography**: I can provide information on countries, "
                        "cultures, travel tips, and help plan trips.\\\\n7. **Writing and Editing**: I can help with "
                        "writing tips, grammar checks, and suggestions for improving your text.\\\\n8. **Health and "
                        "Fitness Advice**: I can offer general advice on exercise, "
                        "diet, and wellness, though I recommend"
                        " consulting professionals for personalized guidance.\\\\n9. **News and Current Events**: "
                        "I can provide summaries or explanations of recent happenings around the world, "
                        "up to my last "
                        "update.\\\\n\\\\nAnd much more! If you have a specific question\\\",\\n        "
                        "\\\"role\\\": "
                        "\\\"assistant\\\",\\n        \\\"function_call\\\": null,\\n        "
                        "\\\"tool_calls\\\": null\\n "
                        "     },\\n      \\\"content_filter_results\\\": {\\n        \\\"hate\\\": {\\n          "
                        "\\\"filtered\\\": false,\\n          \\\"severity\\\": \\\"safe\\\"\\n        },\\n        "
                        "\\\"self_harm\\\": {\\n          \\\"filtered\\\": false,\\n          \\\"severity\\\": "
                        "\\\"safe\\\"\\n        },\\n        \\\"sexual\\\": {\\n          "
                        "\\\"filtered\\\": false,\\n "
                        "         \\\"severity\\\": \\\"safe\\\"\\n        },\\n        "
                        "\\\"violence\\\": {\\n          "
                        "\\\"filtered\\\": false,\\n          \\\"severity\\\": "
                        "\\\"safe\\\"\\n        }\\n      }\\n    "
                        "}\\n  ],\\n  \\\"created\\\": 1712626943,\\n  \\\"model\\\": "
                        "\\\"gpt-4\\\",\\n  \\\"object\\\": "
                        "\\\"chat.completion\\\",\\n  \\\"system_fingerprint\\\": "
                        "\\\"fp_2f57f81c11\\\",\\n  \\\"usage\\\":"
                        " {\\n    \\\"completion_tokens\\\": 256,\\n    "
                        "\\\"prompt_tokens\\\": 40,\\n    \\\"total_tokens\\\""
                        ": 296\\n  },\\n  \\\"prompt_filter_results\\\": "
                        "[\\n    {\\n      \\\"prompt_index\\\": 0,\\n      "
                        "\\\"content_filter_results\\\": {\\n        "
                        "\\\"hate\\\": {\\n          \\\"filtered\\\": false,\\n "
                        "         \\\"severity\\\": \\\"safe\\\"\\n        },\\n        "
                        "\\\"self_harm\\\": {\\n          "
                        "\\\"filtered\\\": false,\\n          "
                        "\\\"severity\\\": \\\"safe\\\"\\n        },\\n        "
                        "\\\"sexual\\\": {\\n          \\\"filtered\\\": false,\\n          "
                        "\\\"severity\\\": \\\"safe\\\"\\n "
                        "       },\\n        \\\"violence\\\": {\\n          \\\"filtered\\\": false,\\n          "
                        "\\\"severity\\\": \\\"safe\\\"\\n        }\\n      }\\n    }\\n  ]\\n}\"}}]"),
                    "{\n  \"model\": \"gpt-4-0405\",\n  \"messages\": [\n    {\n      \"role\": \"system\",\n      "
                    "\"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      "
                    "\"content\": \"hello\"\n    },\n    {\n      \"role\": \"assistant\",\n      \"content\": "
                    "\"Hello! How can I assist you today?\"\n    },\n    {\n      \"role\": \"user\",\n      "
                    "\"content\": \"what can you do?\"\n    }\n  ],\n  \"temperature\": 0.7,\n  \"top_p\": 1.0,\n  "
                    "\"n\": 1,\n  \"stream\": false,\n  \"presence_penalty\": 0.0,\n  \"frequency_penalty\": 0.0,\n  "
                    "\"user\": \"\",\n  \"max_tokens\": 256\n}",
                    "{\n  \"id\": \"chatcmpl-9Bun9MD3qDvYLvVeGenY4uF6sLyS1\",\n  \"choices\": [\n    {\n      "
                    "\"finish_reason\": \"length\",\n      \"index\": 0,\n      \"logprobs\": null,\n      "
                    "\"message\": "
                    "{\n        \"content\": \"I can help with a wide range of questions and tasks! "
                    "Here are some examples:"
                    "\\n\\n1. **General Knowledge and Facts**: "
                    "I can provide information on a vast array of topics such as "
                    "history, science, technology, and more.\\n2. "
                    "**Educational Assistance**: I can help with explanations, "
                    "summaries, and information on various academic subjects.\\n3. "
                    "**Daily Tasks**: I can assist with setting "
                    "reminders, providing weather updates, or offering advice "
                    "on daily routines.\\n4. **Technical Support**: "
                    "I can offer guidance on using software, "
                    "troubleshooting basic tech issues, or understanding tech concepts."
                    "\\n5. **Entertainment and Recommendations**: "
                    "I can suggest books, movies, music, or games based on your "
                    "preferences.\\n6. **Travel and Geography**: "
                    "I can provide information on countries, cultures, travel tips, "
                    "and help plan trips.\\n7. **Writing and Editing**: "
                    "I can help with writing tips, grammar checks, and "
                    "suggestions for improving your text.\\n8. **Health "
                    "and Fitness Advice**: I can offer general advice on "
                    "exercise, diet, and wellness, though I recommend "
                    "consulting professionals for personalized guidance.\\n9. "
                    "**News and Current Events**: I can provide summaries or "
                    "explanations of recent happenings around the world, "
                    "up to my last update.\\n\\nAnd much more! If you have a specific question\",\n"
                    "        \"role\": \"assistant\",\n"
                    "        \"function_call\": null,\n        \"tool_calls\": null\n      },\n"
                    "      \"content_filter_results\": {\n "
                    "       \"hate\": {\n          \"filtered\": false,\n          "
                    "\"severity\": \"safe\"\n        },\n        "
                    "\"self_harm\": {\n          \"filtered\": false,\n          "
                    "\"severity\": \"safe\"\n        },\n        \"sexual\": {\n "
                    "         \"filtered\": false,\n          \"severity\": "
                    "\"safe\"\n        },\n        \"violence\": {\n          "
                    "\"filtered\": false,\n          \"severity\": \"safe\"\n        "
                    "}\n      }\n    }\n  ],\n  \"created\": 1712626943,\n"
                    "  \"model\": \"gpt-4\",\n  \"object\": \"chat.completion\",\n  "
                    "\"system_fingerprint\": \"fp_2f57f81c11\",\n  \"usage\": "
                    "{\n    \"completion_tokens\": 256,\n    \"prompt_tokens\": 40,\n    "
                    "\"total_tokens\": 296\n  },\n  \"prompt_filter_results\":"\
                    " [\n    {\n      \"prompt_index\": 0,\n      \"content_filter_results\": {\n        "
                    "\"hate\": {\n          \"filtered\": false,\n"
                    "          \"severity\": \"safe\"\n        },\n        \"self_harm\": {\n          "
                    "\"filtered\": false,\n          "
                    "\"severity\": \"safe\"\n        },\n        \"sexual\": {\n          "
                    "\"filtered\": false,\n          \"severity\": \"safe\"\n"
                    "        },\n        \"violence\": {\n          \"filtered\": false,\n          "
                    "\"severity\": \"safe\"\n        }\n      }\n    }\n  ]\n}"),
                (
                    Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0), trace_id="1", status="OK",
                        attributes="{\"inputs\":\"{\\n  \\\"model\\\": \\\"gpt-4-0405\\\",\\n  \\\"messages\\\":"
                        "[\\n    {\\n      \\\"role\\\": \\\"system\\\",\\n      \\\"content\\\": \\\"You are a help"
                        "ful assistant.\\\"\\n    },\\n    {\\n      \\\"role\\\": "
                        "\\\"user\\\",\\n      \\\"content\\\": "
                        "\\\"hello\\\"\\n    },\\n    {\\n      \\\"role\\\": \\\"assistant\\\",\\n      "
                        "\\\"content\\\": "
                        "\\\"Hello! How can I assist you today?\\\"\\n    },\\n    {\\n      \\\"role\\\": "
                        "\\\"user\\\",\\n "
                        "     \\\"content\\\": \\\"what can you do?\\\"\\n    }\\n  ],\\n  "
                        "\\\"temperature\\\": 0.7,\\n  "
                        "\\\"top_p\\\": 1.0,\\n  \\\"n\\\": 1,\\n  \\\"stream\\\": false,\\n  "
                        "\\\"presence_penalty\\\": 0.0,\\n "
                        " \\\"frequency_penalty\\\": 0.0,\\n  \\\"user\\\": \\\"\\\",\\n  "
                        "\\\"max_tokens\\\": 256\\n}\"}",
                        events="[]"),
                    "{\n  \"model\": \"gpt-4-0405\",\n  \"messages\":[\n    {\n      \"role\": \"system\",\n      "
                    "\"content\": \"You are a helpful assistant.\"\n    },\n    {\n      \"role\": \"user\",\n      "
                    "\"content\": \"hello\"\n    },\n    {\n      \"role\": \"assistant\",\n      \"content\": "
                    "\"Hello! How can I assist you today?\"\n    },\n    {\n      \"role\": \"user\",\n      "
                    "\"content\": \"what can you do?\"\n    }\n  ],\n  \"temperature\": 0.7,\n  \"top_p\": 1.0,\n  "
                    "\"n\": 1,\n  \"stream\": false,\n  \"presence_penalty\": 0.0,\n  \"frequency_penalty\": 0.0,\n  "
                    "\"user\": \"\",\n  \"max_tokens\": 256\n}",
                    None),
                (
                    Row(span_id="0", parent_id=None, start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0), trace_id="1", status="OK",
                        attributes="{\"inputs\":\"{\\n  \\\"model\\\": \\\"gpt-4-0405\\\",\\n  \\\"messages\\\":"
                        "[\\n    {\\n      \\\"role\\\": \\\"system\\\",\\n      \\\"content\\\": \\\"You are a help"
                        "ful assistant.\\\"\\n    },\\n    {\\n      \\\"role\\\": "
                        "\\\"user\\\",\\n      \\\"content\\\": "
                        "\\\"hello\\\"\\n    },\\n    {\\n      \\\"role\\\": \\\"assistant\\\",\\n      "
                        "\\\"content\\\": "
                        "\\\"Hello! How can I assist you today?\\\"\\n    },\\n    {\\n      "
                        "\\\"role\\\": \\\"user\\\",\\n "
                        "     \\\"content\\\": \\\"what can you do?\\\"\\n    }\\n  ],\\n  "
                        "\\\"temperature\\\": 0.7,\\n  "
                        "\\\"top_p\\\": 1.0,\\n  \\\"n\\\": 1,\\n  \\\"stream\\\": false,\\n  "
                        "\\\"presence_penalty\\\": 0.0,\\n "
                        " \\\"frequency_penalty\\\": 0.0,\\n  \\\"user\\\": \\\"\\\",\\n  "
                        "\\\"max_tokens\\\": 256\\n}\"}",
                        events="[{\"name\":\"promptflow.function.inputs\",\"timestamp\":"
                        "\"2024-04-09T01:42:23.587010Z\","
                        "\"attributes\":{\"payload\":\"{\\n  \\\"model\\\": \\\"gpt-4-0405\\\"\\n  }\"}}]"),
                    "{\n  \"model\": \"gpt-4-0405\"\n  }",
                    None),
                ])
    def test_span_tree_node_input_output_properties(self, row: Row, inputs, output):
        """Test scenarios for getting the input and output tree node properties."""
        node = SpanTreeNode(row)
        assert node.input == inputs
        assert node.output == output

    @pytest.mark.parametrize(
            "row,retrieval_query,retrieval_document",
            [
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        span_type="Retrieval",
                        attributes=None,
                        events=None),
                    None,
                    None),
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        span_type="Retrieval",
                        attributes="{\"span_type\":\"Retrieval\",\"retrieval.query\":"
                        "\"Can you provide information about AML?\",\"retrieval.documents\":"
                        "\"[{\\\"document.content\\\": \\\"# Monitoring Azure Machine "
                        "Learning data reference\\\"}]\"}",
                        events=""),
                    "Can you provide information about AML?",
                    "[{\"document.content\": \"# Monitoring Azure Machine Learning data reference\"}]"),
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        span_type="Retrieval",
                        attributes="{\"span_type\":\"Retrieval\",\"retrieval.query\":"
                        "\"Can you provide information about AML?\","
                        "\"retrieval.documents\":\"[{\\\"document.content\\\": "
                        "\\\"# Monitoring Azure Machine Learning data reference\\\"}]\"}",
                        events="[\n  {\"name\":\"promptflow.retrieval.query\",\n  "
                        "\"timestamp\":\"2024-04-10T15:26:11.410541Z\",\n"
                        "  \"attributes\":{\"payload\":\"Can you provide information about AML?\"}},\n  "
                        "{\"name\":\"promptflow.retrieval.documents\",\n  "
                        "\"timestamp\":\"2024-04-10T15:26:18.429217Z\",\n  "
                        "\"attributes\":{\"payload\":\"[{\\\"document.content\\\": "
                        "\\\"# Monitoring Azure Machine Learning data "
                        "reference\\\"}]\"}}\n  ]"),
                    "Can you provide information about AML?",
                    "[{\"document.content\": \"# Monitoring Azure Machine Learning data reference\"}]"),
            ])
    def test_span_tree_node_retrieval_properties(self, row: Row, retrieval_query, retrieval_document):
        """Test scenarios for getting the input and output tree node properties."""
        node = SpanTreeNode(row)
        assert node.retrieval_query == retrieval_query
        assert node.retrieval_document == retrieval_document

    @pytest.mark.parametrize(
            "row,embeddings",
            [
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        attributes=None,
                        events=None),
                    None),
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        span_type="Embedding",
                        attributes="{\"span_type\": \"Embedding\"}",
                        events="[        {\n          \"name\": \"promptflow.function.inputs\",\n "
                        "         \"timestamp\": \"2024-04-10T15:25:48.909821Z\",\n        "
                        "  \"attributes\": {\n            \"payload\": \"{\\n  \\\"input\\\": [\\n"
                        "    [\\n      4438,\\n      527,\\n      499,\\n      3815,\\n      30\\n "
                        "   ]\\n  ],\\n  \\\"model\\\": \\\"text-embedding-ada-002\\\"\\n}\"\n      "
                        "    }\n        },\n        {\n          \"name\": \"promptflow.embedding.embeddings\","
                        "\n          \"timestamp\": \"2024-04-10T15:25:49.146844Z\",\n          "
                        "\"attributes\": {\n            \"payload\": \"[\\n  {\\n    "
                        "\\\"embedding.vector\\\": \\\"\\u003c1536 dimensional vector\\u003e\\\",\\n    "
                        "\\\"embedding.text\\\": \\\"\\u003c5 dimensional token\\u003e\\\"\\n  }\\n]\"\n"
                        "}\n        }]"),
                    "[\n  {\n    \"embedding.vector\": \"\u003c1536 dimensional vector\u003e\",\n    "
                    "\"embedding.text\": \"\u003c5 dimensional token\u003e\"\n  }\n]"),
                (
                    Row(span_id="0",
                        parent_id=None,
                        start_time=datetime(2024, 2, 12, 0, 0, 1),
                        end_time=datetime(2024, 2, 12, 1, 40, 0),
                        trace_id="1",
                        status="OK",
                        span_type="Embedding",
                        attributes="{\"span_type\": \"Embedding\",\"embedding.embeddings\": \"[\\n  "
                        "{\\n    \\\"embedding.vector\\\": \\\"\\u003c1536 dimensional vector\\u003e\\\",\\n"
                        "    \\\"embedding.text\\\": \\\"\\u003c5 dimensional token\\u003e\\\"\\n  }\\n]\"}",
                        events="[]"),
                    "[\n  {\n    \"embedding.vector\": \"\u003c1536 dimensional vector\u003e\",\n    "
                    "\"embedding.text\": \"\u003c5 dimensional token\u003e\"\n  }\n]")
            ])
    def test_span_tree_node_embedding_properties(self, row: Row, embeddings):
        """Test scenarios for getting the input and output tree node properties."""
        node = SpanTreeNode(row)
        assert node.embeddings == embeddings
