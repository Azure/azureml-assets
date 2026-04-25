# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Customer Satisfaction Evaluator."""

import os
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from azure.ai.evaluation._exceptions import EvaluationException

from .base_evaluator_behavior_test import BaseEvaluatorBehaviorTest
from ...builtin.customer_satisfaction.evaluator._customer_satisfaction import (
    CustomerSatisfactionEvaluator,
)
from ..common.evaluator_mock_config import get_flow_side_effect_for_evaluator


@pytest.mark.unittest
class TestCustomerSatisfactionEvaluatorBehavior(BaseEvaluatorBehaviorTest):
    """
    Behavioral tests for Customer Satisfaction Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = CustomerSatisfactionEvaluator

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.MINIMAL_RESPONSE

    _additional_expected_field_suffixes = ["status", "properties"]

    @property
    def expected_result_fields(self):
        """Get the expected result fields for customer satisfaction evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_score",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_status",
            f"{self._result_prefix}_properties",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]

    def assert_not_applicable(self, result_data: Dict[str, Any]):
        """Assert that the result is not applicable."""
        assert result_data["score"] is None
        assert result_data["label"] == "not_applicable"
        assert "Not applicable" in result_data.get("reason", "")


def _create_mocked_evaluator():
    """Create a CustomerSatisfactionEvaluator with both _flow and _multi_turn_flow mocked."""
    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://Sanitized.api.cognitive.microsoft.com"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "aoai-deployment"),
    )
    evaluator = CustomerSatisfactionEvaluator(model_config=model_config)
    mock_side_effect = get_flow_side_effect_for_evaluator("customer_satisfaction")
    evaluator._flow = MagicMock(side_effect=mock_side_effect)
    evaluator._multi_turn_flow = MagicMock(side_effect=mock_side_effect)
    return evaluator


# region Session-level (messages) behavioral tests

VALID_MESSAGES: List[Dict[str, Any]] = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "I need to cancel my order #12345."}],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I've cancelled your order. Refund in 3-5 business days."}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Has my other order #12346 shipped?"}],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Yes, it shipped yesterday via FedEx. Tracking: FX123456."}],
    },
]


@pytest.mark.unittest
class TestCustomerSatisfactionSessionBehavior:
    """Behavioral tests for the session-level (messages) path of CustomerSatisfactionEvaluator."""

    def test_messages_valid_input(self):
        """Valid messages list produces expected output fields."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(messages=VALID_MESSAGES)

        assert "customer_satisfaction" in result
        assert "customer_satisfaction_score" in result
        assert "customer_satisfaction_result" in result
        assert "customer_satisfaction_reason" in result
        assert "customer_satisfaction_status" in result
        assert "customer_satisfaction_properties" in result
        assert "customer_satisfaction_threshold" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0
        assert result["customer_satisfaction_score"] == result["customer_satisfaction"]
        assert result["customer_satisfaction_status"] == "completed"
        assert isinstance(result["customer_satisfaction_properties"], dict)

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
            {"role": "system", "content": "You are a helpful customer support agent."},
        ] + VALID_MESSAGES
        result = evaluator(messages=messages_with_system)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

    def test_messages_with_tool_calls(self):
        """Messages containing tool calls and tool results are handled."""
        evaluator = _create_mocked_evaluator()
        messages_with_tools = [
            {"role": "user", "content": [{"type": "text", "text": "What's the status of my order?"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_1",
                        "name": "check_order_status",
                        "arguments": {"order_id": "12345"},
                    }
                ],
            },
            {
                "tool_call_id": "call_1",
                "role": "tool",
                "content": [{"type": "tool_result", "tool_result": {"status": "shipped", "eta": "Thursday"}}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Your order has shipped and will arrive Thursday."}],
            },
        ]
        result = evaluator(messages=messages_with_tools)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

    def test_messages_intermediate_response(self):
        """Messages ending with a function_call are rejected by validation (must contain text)."""
        evaluator = _create_mocked_evaluator()
        intermediate_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Cancel my order."}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "function_call",
                        "name": "cancel_order",
                        "call_id": "call_1",
                        "arguments": {"order_id": "12345"},
                    }
                ],
            },
        ]
        with pytest.raises(EvaluationException, match="must contain text content"):
            evaluator(messages=intermediate_messages)

    def test_messages_string_content(self):
        """Messages with string content (not list) are handled and user text is preserved."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": "I need help with my order."},
            {"role": "assistant", "content": "Your order has been updated!"},
        ]
        result = evaluator(messages=messages)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0
        # Verify user string content is included in the conversation passed to prompty
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("conversation", "")
        assert "I need help with my order." in conversation_text

    def test_query_response_intermediate_returns_not_applicable_schema(self):
        """Intermediate single-turn responses return the standardized not-applicable schema."""
        evaluator = _create_mocked_evaluator()
        result = evaluator(
            query="Cancel my order.",
            response=[
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "function_call",
                            "name": "cancel_order",
                            "tool_call_id": "call_1",
                            "arguments": {"order_id": "12345"},
                        }
                    ],
                }
            ],
        )

        assert result["customer_satisfaction"] is None
        assert result["customer_satisfaction_score"] is None
        assert result["customer_satisfaction_result"] == "not_applicable"
        assert result["customer_satisfaction_status"] == "skipped"
        assert result["customer_satisfaction_reason"].startswith("Not applicable:")
        assert result["customer_satisfaction_properties"] == {}

    def test_messages_uses_multi_turn_flow(self):
        """Verify that the session path calls _multi_turn_flow, not _flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(messages=VALID_MESSAGES)

        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()

    def test_query_response_uses_single_turn_flow(self):
        """Verify that the query/response path still calls _flow, not _multi_turn_flow."""
        evaluator = _create_mocked_evaluator()
        evaluator(query="How do I return an item?", response="You can return within 30 days.")

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

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

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
            {"role": "narrator", "content": [{"type": "text", "text": "The agent paused."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_missing_role_key(self):
        """Messages missing the role key raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"content": [{"type": "text", "text": "No role here"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_no_user_message(self):
        """Messages without any user message raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_no_assistant_message(self):
        """Messages without any assistant message raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_conversation_ending_with_user(self):
        """Messages ending with a user message raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            {"role": "user", "content": [{"type": "text", "text": "One more thing..."}]},
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_rejects_conversation_ending_with_tool(self):
        """Messages ending with a tool message raise validation error."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Check status"}]},
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "tool_call_id": "call_1", "name": "check", "arguments": {}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": [{"type": "tool_result", "tool_result": {"status": "ok"}}],
            },
        ]
        with pytest.raises(EvaluationException):
            evaluator(messages=messages)

    def test_messages_allows_consecutive_user_messages(self):
        """Consecutive user messages are accepted."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "text", "text": "Also, one more thing"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Sure, here you go!"}]},
        ]
        result = evaluator(messages=messages)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

    def test_messages_allows_consecutive_assistant_messages(self):
        """Consecutive assistant messages are accepted."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Let me check..."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Here's the answer!"}]},
        ]
        result = evaluator(messages=messages)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

    def test_messages_allows_developer_role(self):
        """Messages with developer role are accepted."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "developer", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi! How can I help?"}]},
        ]
        result = evaluator(messages=messages)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0


# endregion
