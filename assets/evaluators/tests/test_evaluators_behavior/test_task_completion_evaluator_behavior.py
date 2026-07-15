# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Task Completion Evaluator."""

import asyncio
import pytest
from typing import Any, Dict, List

from azure.ai.evaluation._exceptions import EvaluationException

from .base_tools_evaluator_behavior_test import BaseToolsEvaluatorBehaviorTest
from .base_evaluator_behavior_test import _TurnLevelUtilE2ETests, _MessagesUtilE2ETests
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from .base_validator_unit_test import (
    AgentResponseReformatUnitTests,
    ConversationSerializationUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    MessagesOrQueryResponseUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    ToolDefinitionsValidatorUnitTests,
)
from ...builtin.task_completion.evaluator._task_completion import (
    TaskCompletionEvaluator,
    EvaluationLevel,
    serialize_messages,
)
from ..common.evaluator_mock_config import (
    create_mocked_evaluator,
    run_none_score_not_applicable,
)


@pytest.mark.unittest
class TestTaskCompletionEvaluatorBehavior(
    BaseToolsEvaluatorBehaviorTest, BaseToolEvaluationTest, _TurnLevelUtilE2ETests, _MessagesUtilE2ETests
):
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


def _create_mocked_evaluator():
    """Create a TaskCompletionEvaluator with both _flow and _multi_turn_flow mocked."""
    return create_mocked_evaluator(TaskCompletionEvaluator, "task_completion")


# region Multi-turn (messages) behavioral tests

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
class TestTaskCompletionMultiturnBehavior:
    """Behavioral tests for the multi-turn (messages) path of TaskCompletionEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "task_completion" in result
        assert "task_completion_result" in result
        assert "task_completion_reason" in result
        assert "task_completion_properties" in result
        assert "task_completion_status" in result
        assert "task_completion_threshold" in result
        assert "task_completion_score" in result
        assert "task_completion_passed" in result
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
        """Messages ending with only tool calls (no final text) are accepted (mid-execution guard removed)."""
        evaluator = _create_mocked_evaluator()
        intermediate_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Book a flight."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "name": "search_flights",
                        "tool_call_id": "call_1",
                        "arguments": {"origin": "NYC", "destination": "London"},
                    }
                ],
            },
        ]
        result = evaluator(messages=intermediate_messages)
        assert isinstance(result, dict)

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
        conversation_text = call_kwargs.kwargs.get("messages", "")
        assert "Hello, book me a flight." in conversation_text

    def test_messages_uses_multi_turn_flow(self):
        """Verify that the multi-turn conversation path calls _multi_turn_flow, not _flow."""
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
                "tool_call_id": "req_1",
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

    def test_messages_with_non_dict_items_raises_error(self):
        """Messages list containing non-dict items raises validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            "not a dict",
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_invalid_role(self):
        """Messages with an invalid role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "narrator", "content": [{"type": "text", "text": "The agent responded."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException, match="Invalid role"):
            evaluator(messages=messages)

    def test_messages_rejects_missing_role_key(self):
        """Messages missing 'role' key raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"content": [{"type": "text", "text": "No role here"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException, match="role"):
            evaluator(messages=messages)

    def test_messages_rejects_no_user_message(self):
        """Messages without any 'user' role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        with pytest.raises(EvaluationException, match="user"):
            evaluator(messages=messages)

    def test_messages_rejects_no_assistant_message(self):
        """Messages without any 'assistant' role raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "Anyone there?"}]},
        ]
        with pytest.raises(EvaluationException, match="assistant"):
            evaluator(messages=messages)

    def test_messages_allows_conversation_ending_with_tool(self):
        """Messages ending with a tool message are accepted (mid-execution guard removed to align with SDK)."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "What's the weather?"}]},
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
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "tool_result", "tool_result": {"temp": "14C"}}],
            },
        ]
        result = evaluator(messages=messages)
        assert isinstance(result, dict)

    def test_messages_allows_consecutive_user_messages(self):
        """Consecutive user messages followed by assistant are valid."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Book a flight."}]},
            {"role": "user", "content": [{"type": "text", "text": "Make it business class."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Booked business class flight!"}]},
        ]
        result = evaluator(messages=messages)
        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_allows_consecutive_assistant_messages(self):
        """Consecutive assistant messages are valid."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Tell me about Paris."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Paris is in France."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "It's known for the Eiffel Tower."}]},
        ]
        result = evaluator(messages=messages)
        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)

    def test_messages_allows_developer_role(self):
        """Messages with 'developer' role are accepted."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "developer", "content": "You are a travel assistant."},
            {"role": "user", "content": [{"type": "text", "text": "Book a flight."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Done!"}]},
        ]
        result = evaluator(messages=messages)
        assert "task_completion" in result
        assert result["task_completion"] in (0, 1)


# endregion


# region evaluation_level tests

def _create_mocked_evaluator_with_level(evaluation_level=None):
    """Create a TaskCompletionEvaluator with evaluation_level and mocked flows."""
    return create_mocked_evaluator(
        TaskCompletionEvaluator, "task_completion", evaluation_level=evaluation_level
    )


@pytest.mark.unittest
class TestTaskCompletionEvaluationLevel:
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
        assert "task_completion" in result

    def test_forced_turn_with_query_response(self):
        """Forced turn level works with query/response."""
        evaluator = _create_mocked_evaluator_with_level(
            evaluation_level=EvaluationLevel.TURN
        )
        result = evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "task_completion" in result

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
        assert "task_completion" in result

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
        assert "Book a flight from NYC to London for next Friday." in query_text
        assert "Yes, book it." in query_text
        assert "Done! Your flight is booked. Confirmation sent." not in query_text
        assert "Done! Your flight is booked. Confirmation sent." in response_text
        assert "task_completion" in result

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
        assert "task_completion" in result

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
        with pytest.raises(EvaluationException, match="Response list cannot be empty"):
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
        assert "task_completion" in result

    def test_string_level_turn(self):
        """String 'turn' is accepted as evaluation_level."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="turn")
        result = evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "task_completion" in result

    def test_empty_string_level_defaults_to_auto_detect_messages(self):
        """Empty string evaluation_level is treated as None (auto-detect) and uses multi-turn for messages."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="")
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        assert "task_completion" in result

    def test_empty_string_level_defaults_to_auto_detect_query_response(self):
        """Empty string evaluation_level is treated as None (auto-detect) and uses single-turn for query/response."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="")
        evaluator(query="Plan a trip.", response="Here's your itinerary.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

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


class TestSerializeMessages:
    """Unit tests for the serialize_messages helper."""

    def test_consecutive_user_messages_grouped_into_one_turn(self):
        """Multiple consecutive user messages should be collected into a single user turn."""
        messages = [
            {"role": "user", "content": "Hello, I need help."},
            {"role": "user", "content": "I am trying to book a flight."},
            {"role": "user", "content": "From London to Paris, next Monday."},
            {"role": "assistant", "content": "Sure! Let me look that up for you."},
        ]
        expected = (
            "User turn 1:\n"
            "  Hello, I need help.\n"
            "  I am trying to book a flight.\n"
            "  From London to Paris, next Monday.\n"
            "\n"
            "Agent turn 1:\n"
            "  Sure! Let me look that up for you."
        )
        assert serialize_messages(messages) == expected

    def test_consecutive_assistant_text_messages_grouped_into_one_agent_turn(self):
        """Multiple consecutive assistant text messages should be collected into a single agent turn."""
        messages = [
            {"role": "user", "content": "Summarize the report."},
            {"role": "assistant", "content": "The report covers three topics."},
            {"role": "assistant", "content": "First, it discusses market trends."},
            {"role": "assistant", "content": "Second, it covers financial performance."},
        ]
        expected = (
            "User turn 1:\n"
            "  Summarize the report.\n"
            "\n"
            "Agent turn 1:\n"
            "  The report covers three topics.\n"
            "  First, it discusses market trends.\n"
            "  Second, it covers financial performance."
        )
        assert serialize_messages(messages) == expected

    def test_alternating_consecutive_user_and_assistant_messages(self):
        """Consecutive user messages then consecutive assistant messages across two exchanges."""
        messages = [
            {"role": "user", "content": "Step 1: set up the environment."},
            {"role": "user", "content": "Step 2: install the dependencies."},
            {"role": "assistant", "content": "Environment set up successfully."},
            {"role": "assistant", "content": "Dependencies installed."},
            {"role": "user", "content": "Now run the tests."},
            {"role": "assistant", "content": "All tests passed."},
        ]
        expected = (
            "User turn 1:\n"
            "  Step 1: set up the environment.\n"
            "  Step 2: install the dependencies.\n"
            "\n"
            "Agent turn 1:\n"
            "  Environment set up successfully.\n"
            "  Dependencies installed.\n"
            "\n"
            "User turn 2:\n"
            "  Now run the tests.\n"
            "\n"
            "Agent turn 2:\n"
            "  All tests passed."
        )
        assert serialize_messages(messages) == expected

    def test_complex_pairing_order_not_mixed_up(self):
        """Three full exchanges with consecutive bursts + a tool call.

        Verifies that the entire serialized transcript matches the expected output exactly,
        ensuring turn pairing is never crossed and ordering is preserved end-to-end.
        """
        messages = [
            # User burst 1 (two messages)
            {"role": "user", "content": "What is the weather in Paris?"},
            {"role": "user", "content": "And also in London?"},
            # Agent burst 1 (two text messages)
            {"role": "assistant", "content": "Paris is sunny and 22 C."},
            {"role": "assistant", "content": "London is cloudy and 15 C."},
            # User burst 2 (two messages)
            {"role": "user", "content": "Which city is warmer?"},
            {"role": "user", "content": "By how many degrees?"},
            # Agent burst 2 — tool call + tool result + final text
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "name": "compare_temps",
                        "arguments": {"city1": "Paris", "city2": "London"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": [{"type": "tool_result", "tool_result": "Paris is warmer by 7 degrees."}],
            },
            {"role": "assistant", "content": "Paris is warmer by 7 degrees."},
            # User burst 3 (single message)
            {"role": "user", "content": "Thanks for the info!"},
            # Agent burst 3 (single text message)
            {"role": "assistant", "content": "You are welcome!"},
        ]
        expected = (
            "User turn 1:\n"
            "  What is the weather in Paris?\n"
            "  And also in London?\n"
            "\n"
            "Agent turn 1:\n"
            "  Paris is sunny and 22 C.\n"
            "  London is cloudy and 15 C.\n"
            "\n"
            "User turn 2:\n"
            "  Which city is warmer?\n"
            "  By how many degrees?\n"
            "\n"
            "Agent turn 2:\n"
            '  [TOOL_CALL] compare_temps(city1="Paris", city2="London")\n'
            "  [TOOL_RESULT] Paris is warmer by 7 degrees.\n"
            "  Paris is warmer by 7 degrees.\n"
            "\n"
            "User turn 3:\n"
            "  Thanks for the info!\n"
            "\n"
            "Agent turn 3:\n"
            "  You are welcome!"
        )
        assert serialize_messages(messages) == expected

    def test_system_content_block_list_flattened_to_string(self):
        """System message with content-block list is flattened, not passed as raw list."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = serialize_messages(messages)
        assert "You are a helpful assistant." in result
        assert "SYSTEM_PROMPT:" in result

    def test_developer_content_block_list_flattened_to_string(self):
        """Developer message with content-block list is treated like system."""
        messages = [
            {"role": "developer", "content": [{"type": "text", "text": "Be concise."}]},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = serialize_messages(messages)
        assert "Be concise." in result

    def test_system_string_content_still_works(self):
        """System message with plain string content still works after the fix."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = serialize_messages(messages)
        assert "You are helpful." in result

    def test_user_content_block_list_produces_flat_turns(self):
        """User messages with content-block lists produce correctly flattened turns."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Question one."}]},
            {"role": "assistant", "content": "Answer one."},
        ]
        result = serialize_messages(messages)
        assert "Question one." in result
        assert "User turn 1:" in result

    def test_multi_text_block_user_message(self):
        """User message with multiple text blocks produces flat turn content."""
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Part A."},
                {"type": "text", "text": "Part B."},
            ]},
            {"role": "assistant", "content": "Got it."},
        ]
        result = serialize_messages(messages)
        assert "Part A." in result
        assert "Part B." in result

    def test_system_content_block_multi_text(self):
        """System message with multiple text blocks joins them with newline."""
        messages = [
            {"role": "system", "content": [
                {"type": "text", "text": "Rule 1."},
                {"type": "text", "text": "Rule 2."},
            ]},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = serialize_messages(messages)
        assert "Rule 1." in result
        assert "Rule 2." in result


# endregion


# region None score handling tests


@pytest.mark.unittest
class TestTaskCompletionNoneScoreHandling:
    """Regression tests for None-score handling in _do_eval (math.isnan(None) fix).

    When the flow returns a skipped/None score, _do_eval must return the standardized
    not-applicable result instead of crashing on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow returns not-applicable."""
        run_none_score_not_applicable(
            TaskCompletionEvaluator,
            "task_completion",
            query="Plan a trip to Paris.",
            response="Here is your itinerary.",
        )

    def test_conversation_level_none_score_does_not_crash(self):
        """Conversation-level eval with score=None from _multi_turn_flow returns not-applicable."""
        run_none_score_not_applicable(TaskCompletionEvaluator, "task_completion", messages=VALID_MESSAGES)


# endregion


@pytest.mark.unittest
class TestTaskCompletionValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ToolDefinitionsValidatorUnitTests,
    ConversationSerializationUnitTests,
    MessagesOrQueryResponseUnitTests,
    AgentResponseReformatUnitTests,
):
    """Low-level unit tests for task_completion's repeated validators, utils and methods."""

    evaluator_class = TaskCompletionEvaluator


@pytest.mark.unittest
class TestTaskCompletionDoEvalBranches:
    """Cover task_completion-specific ``_do_eval`` branches not exercised by the shared mixins."""

    def test_missing_query_and_response_raises(self):
        """Raise when either query or response is missing in the turn-level input."""
        evaluator = create_mocked_evaluator(TaskCompletionEvaluator, "task_completion")
        with pytest.raises(EvaluationException):
            asyncio.run(evaluator._do_eval({"query": "only a query"}))
