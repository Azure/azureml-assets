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

    _additional_expected_field_suffixes = ["dimensions"]

    @property
    def expected_result_fields(self):
        """Get the expected result fields for customer satisfaction evaluator."""
        return [
            f"{self._result_prefix}",
            f"{self._result_prefix}_reason",
            f"{self._result_prefix}_threshold",
            f"{self._result_prefix}_result",
            f"{self._result_prefix}_dimensions",
            f"{self._result_prefix}_prompt_tokens",
            f"{self._result_prefix}_completion_tokens",
            f"{self._result_prefix}_total_tokens",
            f"{self._result_prefix}_finish_reason",
            f"{self._result_prefix}_model",
            f"{self._result_prefix}_sample_input",
            f"{self._result_prefix}_sample_output",
        ]


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
        assert "customer_satisfaction_result" in result
        assert "customer_satisfaction_reason" in result
        assert "customer_satisfaction_dimensions" in result
        assert "customer_satisfaction_threshold" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0

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
        """Messages ending with a function_call return not-applicable result."""
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
        result = evaluator(messages=intermediate_messages)

        assert result["customer_satisfaction_reason"].startswith("Not applicable")

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

    def test_messages_with_non_dict_items(self):
        """Messages list containing non-dict items are skipped gracefully."""
        evaluator = _create_mocked_evaluator()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            "not a dict",
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]
        result = evaluator(messages=messages)

        assert "customer_satisfaction" in result
        assert 1.0 <= result["customer_satisfaction"] <= 5.0


# endregion
