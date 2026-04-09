# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Completion Evaluator."""

import os
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.task_completion.evaluator._task_completion import (
    TaskCompletionEvaluator,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


@pytest.mark.unittest
class TestTaskCompletionEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Task Completion Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.LOCAL_CALLS_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.CODE_INTERPRETER_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_GROUNDING_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FILE_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.AZURE_AI_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.SHAREPOINT_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.FABRIC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.OPENAPI_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.WEB_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.BROWSER_AUTOMATION_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.IMAGE_GEN_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MEMORY_SEARCH_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.KB_MCP_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "response": data.MCP_EXPECTED_FLOW_RESPONSE,
        "tool_definitions": data.MCP_TC_EXPECTED_FLOW_TOOL_DEFINITIONS_STR,
    }
    # endregion

    evaluator_type = TaskCompletionEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    _additional_expected_field_suffixes = ["details"]


def _create_mocked_evaluator():
    """Create a TaskCompletionEvaluator with both _flow and _multi_turn_flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = TaskCompletionEvaluator(model_config=model_config)
    mock_side_effect = get_flow_side_effect_for_evaluator("task_completion")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=mock_side_effect)
    return evaluator


# region Session-level (messages) behavioral tests

VALID_MESSAGES: List[Dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "Book a flight from NYC to London for next Friday."}],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I found a direct British Airways flight at $450. Shall I book it?"}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Yes, book it."}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Done! Your flight is booked. Confirmation sent."}],
    },
]

VALID_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "search_flights",
        "description": "Search for flights between two cities.",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
            },
        },
    }
]


@pytest.mark.unittest
class TestTaskCompletionSessionBehavior:
    """Behavioral tests for the session-level (messages) path of TaskCompletionEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "task_completion" in result
        assert "task_completion_result" in result
        assert "task_completion_reason" in result
        assert "task_completion_details" in result
        assert "task_completion_threshold" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_with_tool_definitions(self):
        """Messages plus tool_definitions works correctly."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_empty_list_raises_error(self):
        """Empty messages list raises validation error."""
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_messages_invalid_type_raises_error(self):
        """Non-list messages raises validation error."""
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages="not a list")

    def test_messages_with_system_message(self):
        """Messages with a system message are handled correctly."""
        evaluator = _create_mocked_evaluator()
        messages_with_system = [
            {"role": "system", "content": "You are a helpful travel assistant."},
        ] + VALID_MESSAGES
        result = evaluator(messages=messages_with_system)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_with_tool_calls(self):
        """Messages containing tool calls and tool results are handled."""
        evaluator = _create_mocked_evaluator()
        messages_with_tools = [
            {"role": "user", "content": [{"type": "text", "text": "What's the weather in Seattle?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "get_weather",
                        "arguments": {"city": "Seattle"},
                    }
                ],
            },
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": [{"type": "tool_result", "tool_result": {"weather": "Rainy, 14°C"}}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The weather in Seattle is rainy at 14°C."}],
            },
        ]
        result = evaluator(messages=messages_with_tools)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_intermediate_response(self):
        """Messages ending with a function_call return not-applicable result."""
        evaluator = _create_mocked_evaluator()
        intermediate_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Book a flight."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "name": "search_flights",
                        "call_id": "call_1",
                        "arguments": {"origin": "NYC", "destination": "London"},
                    }
                ],
            },
        ]
        result = evaluator(messages=intermediate_messages)

        assert result["task_completion_reason"].startswith("Not applicable")

    def test_messages_string_content(self):
        """Messages with string content (not list) are handled and user text is preserved."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": "Hello, book me a flight."},
            {"role": "assistant", "content": "Your flight is booked!"},
        ]
        result = evaluator(messages=messages)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)
        # Verify user string content is included in the conversation passed to prompty
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("conversation", "")
        assert "Hello, book me a flight." in conversation_text

    def test_messages_uses_multi_turn_flow(self):
        """Verify that the session path calls _multi_turn_flow, not _flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(messages=VALID_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_query_response_uses_single_turn_flow(self):
        """Verify that the query/response path still calls _flow, not _multi_turn_flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(query="Plan a trip to Paris.", response="Here's your itinerary...")

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_with_mcp_approval(self):
        """MCP approval messages are dropped during preprocessing."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Do something"}]},
            {
                "role": "assistant",
                "content": [{"type": "mcp_approval_request", "id": "req_1"}],
            },
            {
                "role": "tool",
                "content": [{"type": "mcp_approval_response", "id": "req_1", "approved": True}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Done!"}]},
        ]
        result = evaluator(messages=messages)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_without_tool_definitions(self):
        """Messages without tool_definitions still works correctly."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)
        # Verify tool_definitions was NOT passed to the prompty
        call_kwargs = evaluator._multi_turn_flow.call_args
        assert "tool_definitions" not in call_kwargs.kwargs

    def test_messages_with_invalid_tool_definitions_type(self):
        """Messages with non-list/non-string tool_definitions raises validation error."""
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=VALID_MESSAGES, tool_definitions=12345)

    def test_messages_with_empty_tool_definitions(self):
        """Messages with empty tool_definitions list is treated as no tool_definitions (optional)."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES, tool_definitions=[])

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_with_invalid_tool_definitions_missing_name(self):
        """Messages with tool_definitions missing 'name' raises validation error."""
        evaluator = _create_mocked_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(
                messages=VALID_MESSAGES,
                tool_definitions=[{"description": "A tool", "parameters": {"type": "object", "properties": {}}}],
            )

    def test_messages_with_non_dict_items(self):
        """Messages list containing non-dict items are skipped gracefully."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            "not a dict",
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]
        result = evaluator(messages=messages)

        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)


# endregion
