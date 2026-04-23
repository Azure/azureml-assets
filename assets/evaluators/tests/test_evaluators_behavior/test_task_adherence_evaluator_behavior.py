# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Adherence Evaluator."""

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.task_adherence.evaluator._task_adherence import (
    TaskAdherenceEvaluator,
    EvaluationLevel,
    serialize_messages,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


@pytest.mark.unittest
class TestTaskAdherenceEvaluatorBehavior(BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Task Adherence Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "system_message": "",
        "query": data.LOCAL_CALLS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_code_interpreter_expected_flow_inputs = {
        "system_message": "",
        "query": data.CODE_INTERPRETER_EXPECTED_FLOW_QUERY,
        "response": data.CODE_INTERPRETER_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_bing_grounding_expected_flow_inputs = {
        "system_message": "",
        "query": data.BING_GROUNDING_EXPECTED_FLOW_QUERY,
        "response": data.BING_GROUNDING_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_bing_custom_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.BING_CUSTOM_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_file_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.FILE_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_azure_ai_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.AZURE_AI_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.AZURE_AI_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "system_message": "",
        "query": data.SHAREPOINT_EXPECTED_FLOW_QUERY,
        "response": data.SHAREPOINT_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "system_message": "",
        "query": data.FABRIC_EXPECTED_FLOW_QUERY,
        "response": data.FABRIC_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_openapi_expected_flow_inputs = {
        "system_message": "",
        "query": data.OPENAPI_EXPECTED_FLOW_QUERY,
        "response": data.OPENAPI_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_web_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.WEB_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.WEB_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_browser_automation_expected_flow_inputs = {
        "system_message": "",
        "query": data.BROWSER_AUTOMATION_EXPECTED_FLOW_QUERY,
        "response": data.BROWSER_AUTOMATION_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_image_generation_expected_flow_inputs = {
        "system_message": "",
        "query": data.IMAGE_GEN_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_memory_search_expected_flow_inputs = {
        "system_message": "",
        "query": data.MEMORY_SEARCH_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_kb_mcp_expected_flow_inputs = {
        "system_message": "",
        "query": data.KB_MCP_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }

    test_mcp_expected_flow_inputs = {
        "system_message": "",
        "query": data.MCP_EXPECTED_FLOW_QUERY,
        "response": data.MCP_EXPECTED_FLOW_RESPONSE,
        "tool_calls": "",
    }
    # endregion

    evaluator_type = TaskAdherenceEvaluator

    MINIMAL_RESPONSE = BaseToolsEvaluatorBehaviorTest.email_tool_call_and_assistant_response

    _additional_expected_field_suffixes = ["details", "properties"]

def _create_mocked_evaluator():
    """Create a TaskAdherenceEvaluator with both _flow and _multi_turn_flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = TaskAdherenceEvaluator(model_config=model_config)
    mock_side_effect = get_flow_side_effect_for_evaluator("task_adherence")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=mock_side_effect)
    return evaluator


# region Conversation-level (messages) behavioral tests

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
class TestTaskAdherenceMultiturnBehavior:
    """Behavioral tests for the multi-turn (messages) path of TaskAdherenceEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "task_adherence" in result
        assert "task_adherence_result" in result
        assert "task_adherence_reason" in result
        assert "task_adherence_details" in result
        assert "task_adherence_properties" in result
        assert "task_adherence_threshold" in result
        assert result["task_adherence"] in (0.0, 1.0)

    def test_messages_with_tool_definitions(self):
        """Messages plus tool_definitions works correctly."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES, tool_definitions=VALID_TOOL_DEFINITIONS)

        assert "task_adherence" in result
        assert result["task_adherence"] in (0.0, 1.0)

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

        assert "task_adherence" in result
        assert result["task_adherence"] in (0.0, 1.0)

    def test_messages_uses_multi_turn_flow(self):
        """Messages input calls _multi_turn_flow and not _flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(messages=VALID_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_query_response_uses_single_turn_flow(self):
        """query/response input still calls _flow and not _multi_turn_flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(query="Plan a trip.", response="Here's your itinerary.")

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_without_tool_definitions(self):
        """Messages path does not inject tool_definitions when absent."""
        evaluator = _create_mocked_evaluator()
        evaluator(messages=VALID_MESSAGES)

        call_kwargs = evaluator._multi_turn_flow.call_args
        assert "tool_definitions" not in call_kwargs.kwargs

    def test_messages_rejects_invalid_role(self):
        """Messages with invalid role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "narrator", "content": [{"type": "text", "text": "The agent answered."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]
        with pytest.raises(EvaluationException, match="Invalid role"):
            evaluator(messages=messages)

    def test_messages_rejects_no_user_message(self):
        """Messages without user role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
        with pytest.raises(EvaluationException, match="user"):
            evaluator(messages=messages)

    def test_messages_rejects_no_assistant_message(self):
        """Messages without assistant role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        with pytest.raises(EvaluationException, match="assistant"):
            evaluator(messages=messages)

    def test_messages_rejects_conversation_ending_with_user(self):
        """Messages ending with user raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "user", "content": [{"type": "text", "text": "Thanks"}]},
        ]
        with pytest.raises(EvaluationException, match="last message must have role 'assistant'"):
            evaluator(messages=messages)

    def test_messages_intermediate_response(self):
        """Messages ending with only tool calls are rejected."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Search for info."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "name": "search_info",
                        "tool_call_id": "call_1",
                        "arguments": {"query": "info"},
                    }
                ],
            },
        ]
        with pytest.raises(EvaluationException, match="must contain text content"):
            evaluator(messages=messages)

# endregion


# region evaluation_level tests

def _create_mocked_evaluator_with_level(evaluation_level=None):
    """Create a TaskAdherenceEvaluator with evaluation_level and mocked flows."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = TaskAdherenceEvaluator(
        model_config=model_config,
        evaluation_level=evaluation_level,
    )
    mock_side_effect = get_flow_side_effect_for_evaluator("task_adherence")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=mock_side_effect)
    return evaluator


@pytest.mark.unittest
class TestTaskAdherenceEvaluationLevel:
    """Tests for the evaluation_level parameter."""

    def test_auto_detect_uses_multi_turn_for_messages(self):
        """Default (None) mode auto-detects multi-turn when messages provided."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level=None)
        evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_auto_detect_uses_single_turn_for_query_response(self):
        """Default (None) mode auto-detects single-turn when query/response provided."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level=None)
        evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_forced_conversation_with_messages(self):
        """Forced conversation level works with messages."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        assert "task_adherence" in result

    def test_forced_turn_with_query_response(self):
        """Forced turn level works with query/response."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.TURN
        )
        result = evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "task_adherence" in result

    def test_forced_conversation_with_query_response_message_lists_converts(self):
        """Forced conversation level converts query/response message lists into messages."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        result = evaluator(query=VALID_MESSAGES[:3], response=VALID_MESSAGES[3:])
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        call_kwargs = evaluator._multi_turn_flow.call_args
        merged_messages = call_kwargs.kwargs.get("messages", "")
        assert "Book a flight from NYC to London for next Friday." in merged_messages
        assert "Yes, book it." in merged_messages
        assert "Done! Your flight is booked. Confirmation sent." in merged_messages
        assert "task_adherence" in result

    def test_forced_turn_with_messages_converts(self):
        """Forced turn level converts messages into query/response around the latest user turn."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.TURN
        )
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        call_kwargs = evaluator._flow.call_args
        query_text = call_kwargs.kwargs.get("query", "")
        response_text = call_kwargs.kwargs.get("response", "")
        assert "Yes, book it." in query_text
        assert "Done! Your flight is booked. Confirmation sent." in response_text
        assert "task_adherence" in result

    def test_forced_conversation_with_string_query_response_wraps_to_messages(self):
        """Forced conversation level wraps string query/response into messages and uses multi-turn."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        result = evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("messages", "")
        assert "Plan a trip." in conversation_text
        assert "Here's your itinerary." in conversation_text
        assert "task_adherence" in result

    def test_forced_conversation_with_empty_string_query_raises(self):
        """Forced conversation level rejects empty string query."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        with pytest.raises(EvaluationException):
            evaluator(query="", response="Here's your itinerary.")

    def test_forced_turn_with_messages_without_response_raises_invalid_value(self):
        """Forced turn level requires response messages after the latest user turn."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.TURN
        )
        with pytest.raises(EvaluationException, match="last message must have role 'assistant'"):
            evaluator(
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": "Book a flight."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "Where to?"}]},
                    {"role": "user", "content": [{"type": "text", "text": "To London."}]},
                ]
            )

    def test_string_level_conversation(self):
        """String 'conversation' is accepted as evaluation_level."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="conversation")
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        assert "task_adherence" in result

    def test_string_level_turn(self):
        """String 'turn' is accepted as evaluation_level."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="turn")
        result = evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "task_adherence" in result

    def test_invalid_string_level_raises(self):
        """Invalid string evaluation_level raises at init time."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _create_mocked_evaluator_with_level(evaluation_level="batch")

    def test_invalid_type_level_raises(self):
        """Non-string/non-enum evaluation_level raises at init time."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _create_mocked_evaluator_with_level(evaluation_level=42)


# endregion


# region serialize_messages tests


@pytest.mark.unittest
class TestTaskAdherenceSerializeMessages:
    """Unit tests for the serialize_messages helper used by task adherence."""

    def test_simple_conversation(self):
        """Simple user/assistant exchange serializes correctly."""
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "The sky is blue."},
        ]
        result = serialize_messages(messages)
        assert "User turn 1:" in result
        assert "What color is the sky?" in result
        assert "Agent turn 1:" in result
        assert "The sky is blue." in result


# endregion
