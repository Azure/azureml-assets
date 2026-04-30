# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Groundedness Evaluator."""

import os
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.groundedness.evaluator._groundedness import (
    GroundednessEvaluator,
    EvaluationLevel,
    serialize_messages,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


@pytest.mark.unittest
class TestGroundednessEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Groundedness Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.LOCAL_CALLS_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.FILE_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.IMAGE_GEN_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.MEMORY_SEARCH_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.KB_MCP_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_GROUNDEDNESS_EXPECTED_FLOW_QUERY,
        "response": data.MCP_GROUNDEDNESS_EXPECTED_FLOW_RESPONSE,
        "context": data.GROUNDEDNESS_NO_CONTEXT,
    }
    # endregion

    evaluator_type = GroundednessEvaluator

    check_for_unsupported_tools = True

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.weather_tool_result_and_assistant_response


# region Conversation-level (messages) behavioral tests


def _create_multi_turn_mock_side_effect():
    """Create a mock side effect that returns dict output for multi-turn groundedness."""

    async def flow_side_effect(timeout, **kwargs):
        return {
            "llm_output": {
                "score": 5,
                "reason": "All responses are grounded in the provided conversation evidence.",
                "status": "completed",
                "properties": {
                    "grounding_sources": ["Tool result"],
                    "grounded_claims": "All claims supported.",
                    "ungrounded_claims": "None",
                },
            }
        }

    return flow_side_effect


def _create_mocked_groundedness_evaluator():
    """Create a GroundednessEvaluator with both _flow and _multi_turn_flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = GroundednessEvaluator(model_config=model_config)
    mock_side_effect = get_flow_side_effect_for_evaluator("groundedness")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=_create_multi_turn_mock_side_effect())
    return evaluator


VALID_GROUNDEDNESS_MESSAGES: List[Dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What are the office hours for the downtown branch?"}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "name": "search_info",
                "arguments": {"query": "downtown branch office hours"},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": [
            {
                "type": "tool_result",
                "tool_result": "Downtown branch: Mon-Fri 9AM-5PM, Sat 10AM-2PM, closed Sunday",
            }
        ],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The downtown branch is open Monday through Friday from 9 AM to 5 PM, "
                "Saturday from 10 AM to 2 PM, and closed on Sundays.",
            }
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Can I visit on Saturday afternoon at 1 PM?"}],
    },
    {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Yes, the downtown branch is open on Saturdays until 2 PM, so a 1 PM visit would work.",
            }
        ],
    },
]


@pytest.mark.unittest
class TestGroundednessMultiturnBehavior:
    """Behavioral tests for the multi-turn (messages) path of GroundednessEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_groundedness_evaluator()
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        assert "groundedness" in result
        assert "groundedness_result" in result
        assert "groundedness_reason" in result
        assert "groundedness_properties" in result
        assert "groundedness_status" in result
        assert "groundedness_threshold" in result
        assert 1 <= result["groundedness"] <= 5

    def test_messages_with_tool_definitions(self):
        """Messages plus tool_definitions works correctly."""
        evaluator = _create_mocked_groundedness_evaluator()
        tool_defs = [
            {
                "name": "search_info",
                "description": "Search for information.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            }
        ]
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES, tool_definitions=tool_defs)

        assert "groundedness" in result
        assert 1 <= result["groundedness"] <= 5

    def test_messages_parses_canonical_schema_skipped_output(self):
        """Canonical skipped output is returned as not_applicable with no score."""
        evaluator = _create_mocked_groundedness_evaluator()

        async def canonical_skipped_output(timeout, **kwargs):
            return {
                "llm_output": {
                    "score": None,
                    "reason": "No agent responses to evaluate for groundedness.",
                    "status": "skipped",
                    "properties": None,
                }
            }

        evaluator._multi_turn_flow = MagicMock(side_effect=canonical_skipped_output)
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        assert result["groundedness"] is None
        assert result["groundedness_result"] == "not_applicable"
        assert result["groundedness_reason"] == "No agent responses to evaluate for groundedness."
        assert result["groundedness_status"] == "skipped"
        assert result["groundedness_properties"] == {}

    def test_messages_invalid_output_returns_error_result(self):
        """Invalid non-dict output returns structured error result instead of raising."""
        evaluator = _create_mocked_groundedness_evaluator()

        async def invalid_output(timeout, **kwargs):
            return {"llm_output": "invalid"}

        evaluator._multi_turn_flow = MagicMock(side_effect=invalid_output)
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        assert result["groundedness"] is None
        assert result["groundedness_result"] == "error"
        assert result["groundedness_reason"] == "Evaluator returned invalid output."
        assert result["groundedness_status"] == "error"
        assert result["groundedness_properties"] == {}

    def test_messages_empty_list_raises_error(self):
        """Empty messages list raises validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages=[])

    def test_messages_invalid_type_raises_error(self):
        """Non-list messages raises validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        with pytest.raises(EvaluationException):
            evaluator(messages="not a list")

    def test_messages_with_system_message(self):
        """Messages with a system message are handled correctly."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
        ] + VALID_GROUNDEDNESS_MESSAGES
        result = evaluator(messages=messages_with_system)

        assert "groundedness" in result
        assert 1 <= result["groundedness"] <= 5

    def test_messages_string_content(self):
        """Messages with string content (not list) are handled."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "user", "content": "What color is the sky?"},
            {"role": "assistant", "content": "The sky is blue."},
        ]
        result = evaluator(messages=messages)

        assert "groundedness" in result
        assert 1 <= result["groundedness"] <= 5

    def test_messages_uses_multi_turn_flow(self):
        """Verify that the multi-turn conversation path calls _multi_turn_flow, not _flow."""
        evaluator = _create_mocked_groundedness_evaluator()
        evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_query_response_uses_single_turn_flow(self):
        """Verify that the query/response/context path still calls _flow, not _multi_turn_flow."""
        evaluator = _create_mocked_groundedness_evaluator()
        evaluator(response="The sky is blue.", context="The sky appears blue due to Rayleigh scattering.")

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_with_mcp_approval(self):
        """MCP approval messages are dropped during preprocessing."""
        evaluator = _create_mocked_groundedness_evaluator()
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

        assert "groundedness" in result
        assert 1 <= result["groundedness"] <= 5

    def test_messages_without_tool_definitions(self):
        """Messages without tool_definitions still works correctly."""
        evaluator = _create_mocked_groundedness_evaluator()
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        assert "groundedness" in result
        # Verify tool_definitions was NOT passed to the prompty
        call_kwargs = evaluator._multi_turn_flow.call_args
        assert "tool_definitions" not in call_kwargs.kwargs

    def test_messages_with_non_dict_items_raises_error(self):
        """Messages list containing non-dict items raises validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            "not a dict",
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_invalid_role(self):
        """Messages with an invalid role raise validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "narrator", "content": [{"type": "text", "text": "The agent responded."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException, match="Invalid role"):
            evaluator(messages=messages)

    def test_messages_rejects_no_user_message(self):
        """Messages without any 'user' role raise validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        with pytest.raises(EvaluationException, match="user"):
            evaluator(messages=messages)

    def test_messages_rejects_no_assistant_message(self):
        """Messages without any 'assistant' role raise validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "Anyone there?"}]},
        ]
        with pytest.raises(EvaluationException, match="assistant"):
            evaluator(messages=messages)

    def test_messages_rejects_conversation_ending_with_user(self):
        """Messages ending with a user message raise validation error."""
        evaluator = _create_mocked_groundedness_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            {"role": "user", "content": [{"type": "text", "text": "Thanks, bye"}]},
        ]
        with pytest.raises(EvaluationException, match="last message must have role 'assistant'"):
            evaluator(messages=messages)

    def test_messages_intermediate_response(self):
        """Messages ending with only tool calls (no text) are rejected."""
        evaluator = _create_mocked_groundedness_evaluator()
        intermediate_messages = [
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
            evaluator(messages=intermediate_messages)

    def test_messages_pass_fail_threshold(self):
        """Score result respects threshold for pass/fail."""
        evaluator = _create_mocked_groundedness_evaluator()
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)

        # Default threshold is 3; mock returns score 5
        assert result["groundedness"] == 5
        assert result["groundedness_result"] == "pass"
        assert result["groundedness_threshold"] == 3


# endregion


# region evaluation_level tests

def _create_mocked_groundedness_evaluator_with_level(evaluation_level=None):
    """Create a GroundednessEvaluator with evaluation_level and mocked flows."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = GroundednessEvaluator(
        model_config=model_config,
        evaluation_level=evaluation_level,
    )
    mock_side_effect = get_flow_side_effect_for_evaluator("groundedness")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=_create_multi_turn_mock_side_effect())
    return evaluator


@pytest.mark.unittest
class TestGroundednessEvaluationLevel:
    """Tests for the evaluation_level parameter."""

    def test_auto_detect_uses_multi_turn_for_messages(self):
        """Default (None) mode auto-detects multi-turn when messages provided."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(evaluation_level=None)
        evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_auto_detect_uses_single_turn_for_response_context(self):
        """Default (None) mode auto-detects single-turn when response/context provided."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(evaluation_level=None)
        evaluator(response="The sky is blue.", context="The sky is blue due to Rayleigh scattering.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_forced_conversation_with_messages(self):
        """Forced conversation level works with messages."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        assert "groundedness" in result

    def test_forced_conversation_with_string_query_response_wraps_to_messages(self):
        """Forced conversation level wraps string query/response into messages and uses multi-turn."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(
            evaluation_level=EvaluationLevel.CONVERSATION
        )
        result = evaluator(
            query="What color is the sky?",
            response="The sky is blue.",
            context="The sky is blue."
        )
        # Note: _flow may be reassigned by _ensure_query_prompty_loaded when query is present,
        # so we only assert that multi_turn_flow was used for conversation-level evaluation.
        evaluator._multi_turn_flow.assert_called_once()
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("messages", "")
        assert "What color is the sky?" in conversation_text
        assert "The sky is blue." in conversation_text
        assert "groundedness" in result

    def test_string_level_conversation(self):
        """String 'conversation' is accepted as evaluation_level."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(evaluation_level="conversation")
        result = evaluator(messages=VALID_GROUNDEDNESS_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        assert "groundedness" in result

    def test_string_level_turn(self):
        """String 'turn' is accepted as evaluation_level."""
        evaluator = _create_mocked_groundedness_evaluator_with_level(evaluation_level="turn")
        evaluator(response="The sky is blue.", context="The sky is blue due to scattering.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_invalid_string_level_raises(self):
        """Invalid string evaluation_level raises at init time."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _create_mocked_groundedness_evaluator_with_level(evaluation_level="batch")

    def test_invalid_type_level_raises(self):
        """Non-string/non-enum evaluation_level raises at init time."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _create_mocked_groundedness_evaluator_with_level(evaluation_level=42)


# endregion


# region serialize_messages tests


@pytest.mark.unittest
class TestGroundednessSerializeMessages:
    """Unit tests for the serialize_messages helper used by groundedness."""

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

    def test_multi_turn_conversation(self):
        """Multi-turn conversation serializes with numbered turns."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]
        result = serialize_messages(messages)
        assert "User turn 1:" in result
        assert "Agent turn 1:" in result
        assert "User turn 2:" in result
        assert "Agent turn 2:" in result

    def test_with_tool_calls(self):
        """Tool calls and results are included in serialization."""
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
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The weather is 14C."}],
            },
        ]
        result = serialize_messages(messages)
        assert "User turn 1:" in result
        assert "What's the weather?" in result
        assert "Agent turn 1:" in result

    def test_empty_messages(self):
        """Empty messages list returns empty string."""
        assert serialize_messages([]) == ""

    def test_system_message_included(self):
        """System message is included in serialization."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        result = serialize_messages(messages)
        assert "You are a helpful assistant." in result


# endregion
