# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Coherence Evaluator."""

import os
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from .base_tool_evaluation_test import BaseToolEvaluationTest
from . import common_tool_test_data as data
from ...builtin.coherence.evaluator._coherence import (
    CoherenceEvaluator,
    EvaluationLevel,
    serialize_messages,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


@pytest.mark.unittest
class TestCoherenceEvaluatorBehavior(BaseEvaluatorBehaviorTest, BaseToolEvaluationTest):
    """
    Behavioral tests for Coherence Evaluator.

    Tests different input formats and scenarios.
    """

    # region Expected flow inputs for each test
    test_function_tool_local_calls_expected_flow_inputs = {
        "query": data.LOCAL_CALLS_QUERY,
        "response": data.LOCAL_CALLS_COHERENCE_EXPECTED_FLOW_RESPONSE,
    }

    test_code_interpreter_expected_flow_inputs = {
        "query": data.CODE_INTERPRETER_QUERY,
        "response": data.CODE_INTERPRETER_RESPONSE,
    }

    test_bing_grounding_expected_flow_inputs = {
        "query": data.BING_GROUNDING_QUERY,
        "response": data.BING_GROUNDING_RESPONSE,
    }

    test_bing_custom_search_expected_flow_inputs = {
        "query": data.BING_CUSTOM_SEARCH_QUERY,
        "response": data.BING_CUSTOM_SEARCH_RESPONSE,
    }

    test_file_search_expected_flow_inputs = {
        "query": data.FILE_SEARCH_QUERY,
        "response": data.FILE_SEARCH_RESPONSE,
    }

    test_azure_ai_search_expected_flow_inputs = {
        "query": data.AZURE_AI_SEARCH_QUERY,
        "response": data.AZURE_AI_SEARCH_RESPONSE,
    }

    test_sharepoint_grounding_expected_flow_inputs = {
        "query": data.SHAREPOINT_QUERY,
        "response": data.SHAREPOINT_RESPONSE,
    }

    test_fabric_data_agent_expected_flow_inputs = {
        "query": data.FABRIC_QUERY,
        "response": data.FABRIC_RESPONSE,
    }

    test_openapi_expected_flow_inputs = {
        "query": data.OPENAPI_QUERY,
        "response": data.OPENAPI_RESPONSE,
    }

    test_web_search_expected_flow_inputs = {
        "query": data.WEB_SEARCH_QUERY,
        "response": data.WEB_SEARCH_RESPONSE,
    }

    test_browser_automation_expected_flow_inputs = {
        "query": data.BROWSER_AUTOMATION_QUERY,
        "response": data.BROWSER_AUTOMATION_RESPONSE,
    }

    test_image_generation_expected_flow_inputs = {
        "query": data.IMAGE_GEN_QUERY,
        "response": data.IMAGE_GEN_RESPONSE,
    }

    test_memory_search_expected_flow_inputs = {
        "query": data.MEMORY_SEARCH_QUERY,
        "response": data.MEMORY_SEARCH_RESPONSE,
    }

    test_kb_mcp_expected_flow_inputs = {
        "query": data.KB_MCP_QUERY,
        "response": data.KB_MCP_TCS_EXPECTED_FLOW_RESPONSE,
    }

    test_mcp_expected_flow_inputs = {
        "query": data.MCP_QUERY,
        "response": data.MCP_TCS_EXPECTED_FLOW_RESPONSE,
    }
    # endregion

    evaluator_type = CoherenceEvaluator


def _create_multi_turn_mock_side_effect(
    score: int = 5,
    status: str = "completed",
    reason: str = "Conversation is coherent overall.",
    properties: Dict[str, Any] = None,
):
    """Create a mock side effect that returns dict output for multi-turn coherence."""
    if properties is None and status == "completed":
        properties = {
            "gating_summary": "User flow mostly on-topic.",
            "conversation_flow_summary": "Agent responses follow context across turns.",
            "agent_coherence_issues": "None",
        }

    async def flow_side_effect(timeout, **kwargs):
        return {
            "llm_output": {
                "score": score if status == "completed" else None,
                "status": status,
                "reason": reason,
                "properties": properties if status == "completed" else None,
            }
        }

    return flow_side_effect


def _create_mocked_coherence_evaluator(evaluation_level=None, multi_turn_side_effect=None):
    """Create a CoherenceEvaluator with both _flow and _multi_turn_flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = CoherenceEvaluator(model_config=model_config, evaluation_level=evaluation_level)
    evaluator._flow = MagicMock(side_effect=get_flow_side_effect_for_evaluator("coherence"))
    evaluator._multi_turn_flow = MagicMock(side_effect=multi_turn_side_effect or _create_multi_turn_mock_side_effect())
    return evaluator


# region Multi-turn (messages) behavioral tests

VALID_MESSAGES: List[Dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "I need to plan a trip to Paris."}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Sure, what dates are you considering?"}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Next weekend. I also want museum recommendations."}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Great. For next weekend, I recommend the Louvre and Musee d'Orsay."}],
    },
]


@pytest.mark.unittest
class TestCoherenceMultiturnBehavior:
    """Behavioral tests for the multi-turn (messages) path of CoherenceEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_coherence_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "coherence" in result
        assert "coherence_result" in result
        assert "coherence_reason" in result
        assert "coherence_score" in result
        assert "coherence_status" in result
        assert "coherence_properties" in result
        assert "coherence_threshold" in result
        assert 1 <= result["coherence"] <= 5
        assert result["coherence_score"] == result["coherence"]
        assert result["coherence_status"] == "completed"

    def test_messages_string_content(self):
        """Messages with string content are handled and serialized."""
        evaluator = _create_mocked_coherence_evaluator()
        messages = [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "It is how plants convert sunlight into energy."},
        ]
        result = evaluator(messages=messages)

        assert "coherence" in result
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("messages", "")
        assert "What is photosynthesis?" in conversation_text

    def test_messages_with_system_message(self):
        """Messages with system/developer context are handled."""
        evaluator = _create_mocked_coherence_evaluator()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
        ] + VALID_MESSAGES
        result = evaluator(messages=messages)
        assert "coherence" in result

    def test_messages_intermediate_response_rejected(self):
        """Messages ending with only tool calls (no text) are rejected."""
        evaluator = _create_mocked_coherence_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Find me flights."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "name": "search_flights",
                        "tool_call_id": "call_1",
                        "arguments": {"origin": "NYC", "destination": "Paris"},
                    }
                ],
            },
        ]
        with pytest.raises(EvaluationException, match="must contain text content"):
            evaluator(messages=messages)

    def test_messages_uses_multi_turn_flow(self):
        """Verify that messages path calls _multi_turn_flow, not _flow."""
        evaluator = _create_mocked_coherence_evaluator()
        evaluator(messages=VALID_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_query_response_uses_single_turn_flow(self):
        """Verify that query/response path still calls _flow."""
        evaluator = _create_mocked_coherence_evaluator()
        evaluator(query="What is photosynthesis?", response="It is how plants convert sunlight into energy.")

        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_messages_skip_output_maps_to_not_applicable(self):
        """Skipped multi-turn output follows standardized skipped schema."""
        skipped_side_effect = _create_multi_turn_mock_side_effect(
            status="skipped",
            reason="Conversation is mostly derailed by unrelated topic jumps.",
        )
        evaluator = _create_mocked_coherence_evaluator(multi_turn_side_effect=skipped_side_effect)
        result = evaluator(messages=VALID_MESSAGES)

        assert result["coherence"] is None
        assert result["coherence_score"] is None
        assert result["coherence_result"] == "skipped"
        assert result["coherence_status"] == "skipped"
        assert result["coherence_properties"] == {}


# endregion


# region evaluation_level tests


@pytest.mark.unittest
class TestCoherenceEvaluationLevel:
    """Tests for the evaluation_level parameter."""

    def test_auto_detect_uses_multi_turn_for_messages(self):
        """Default mode auto-detects multi-turn when messages are provided."""
        evaluator = _create_mocked_coherence_evaluator(evaluation_level=None)
        evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_auto_detect_uses_single_turn_for_query_response(self):
        """Default mode auto-detects single-turn for query/response."""
        evaluator = _create_mocked_coherence_evaluator(evaluation_level=None)
        evaluator(query="What is the capital of France?", response="Paris.")
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()

    def test_forced_conversation_with_string_query_response_wraps_to_messages(self):
        """Forced conversation wraps string query/response into messages and uses multi-turn flow."""
        evaluator = _create_mocked_coherence_evaluator(evaluation_level=EvaluationLevel.CONVERSATION)
        result = evaluator(query="What is the capital of France?", response="Paris.")
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("messages", "")
        assert "What is the capital of France?" in conversation_text
        assert "Paris." in conversation_text
        assert "coherence" in result

    def test_forced_turn_with_messages_converts(self):
        """Forced turn converts messages into query/response and uses single-turn flow."""
        evaluator = _create_mocked_coherence_evaluator(evaluation_level=EvaluationLevel.TURN)
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "coherence" in result

    def test_invalid_evaluation_level_raises(self):
        """Invalid evaluation level raises at init time."""
        with pytest.raises(EvaluationException, match="Invalid evaluation_level"):
            _create_mocked_coherence_evaluator(evaluation_level="batch")


# endregion


# region serialize_messages tests


class TestCoherenceSerializeMessages:
    """Unit tests for coherence serialize_messages helper."""

    def test_simple_conversation_serializes(self):
        """Simple user/assistant messages are serialized with turn labels."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        output = serialize_messages(messages)
        assert "User turn 1:" in output
        assert "Agent turn 1:" in output
        assert "Hello" in output
        assert "Hi there!" in output


# endregion
