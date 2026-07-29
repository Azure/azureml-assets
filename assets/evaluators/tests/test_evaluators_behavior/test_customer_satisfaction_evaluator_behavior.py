# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Customer Satisfaction Evaluator."""

import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from azure.ai.evaluation._exceptions import EvaluationException

from .base_evaluator_behavior_test import (
    BaseEvaluatorBehaviorTest,
    _TurnLevelUtilE2ETests,
    _MessagesUtilE2ETests,
)
from .base_validator_unit_test import (
    AgentResponseReformatUnitTests,
    ConversationSerializationUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ConversationValidatorUnitTests,
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ...builtin.customer_satisfaction.evaluator._customer_satisfaction import (
    CustomerSatisfactionEvaluator,
    serialize_messages,
)
from ..common.evaluator_mock_config import (
    create_mocked_evaluator,
    create_none_score_flow_side_effect,
)


@pytest.mark.unittest
class TestCustomerSatisfactionEvaluatorBehavior(
    BaseEvaluatorBehaviorTest, _TurnLevelUtilE2ETests, _MessagesUtilE2ETests
):
    """
    Behavioral tests for Customer Satisfaction Evaluator.

    Tests different input formats and scenarios.
    """

    evaluator_type = CustomerSatisfactionEvaluator

    MINIMAL_RESPONSE = BaseEvaluatorBehaviorTest.MINIMAL_RESPONSE


def _create_mocked_evaluator():
    """Create a CustomerSatisfactionEvaluator with both _flow and _multi_turn_flow mocked."""
    return create_mocked_evaluator(CustomerSatisfactionEvaluator, "customer_satisfaction")


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
        """Messages ending with a function_call are accepted (mid-execution guard removed to align with SDK)."""
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
        assert isinstance(result, dict)

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
        # properties contains default metadata (all zeros/empty) when no prompty output
        props = result["customer_satisfaction_properties"]
        assert isinstance(props, dict)
        assert props["prompt_tokens"] == 0
        assert props["completion_tokens"] == 0
        assert props["total_tokens"] == 0

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

    def test_messages_allows_conversation_ending_with_tool(self):
        """Messages ending with a tool message are accepted (mid-execution guard removed to align with SDK)."""
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
        result = evaluator(messages=messages)
        assert isinstance(result, dict)

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


# region evaluation_level tests

def _create_mocked_evaluator_with_level(evaluation_level=None):
    """Create a CustomerSatisfactionEvaluator with evaluation_level and mocked flows."""
    return create_mocked_evaluator(
        CustomerSatisfactionEvaluator, "customer_satisfaction", evaluation_level=evaluation_level
    )


@pytest.mark.unittest
class TestCustomerSatisfactionEvaluationLevel:
    """Tests for the evaluation_level parameter."""

    def test_empty_string_level_defaults_to_auto_detect_messages(self):
        """Empty string evaluation_level is treated as None (auto-detect) and uses multi-turn for messages."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="")
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        assert "customer_satisfaction" in result

    def test_empty_string_level_defaults_to_auto_detect_query_response(self):
        """Empty string evaluation_level is treated as None (auto-detect) and uses single-turn for query/response."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="")
        evaluator(query="How do I return an item?", response="You can return within 30 days.")
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

    def test_forced_conversation_with_string_query_response_wraps_to_messages(self):
        """Forced conversation level wraps string query/response into messages and uses multi-turn."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="conversation")
        result = evaluator(query="How do I return an item?", response="You can return within 30 days.")
        evaluator._multi_turn_flow.assert_called_once()
        evaluator._flow.assert_not_called()
        call_kwargs = evaluator._multi_turn_flow.call_args
        conversation_text = call_kwargs.kwargs.get("conversation", "")
        assert "How do I return an item?" in conversation_text
        assert "You can return within 30 days." in conversation_text
        assert "customer_satisfaction" in result

    def test_forced_turn_with_messages_converts(self):
        """Forced turn level converts messages into query/response and uses single-turn flow."""
        evaluator = _create_mocked_evaluator_with_level(evaluation_level="turn")
        result = evaluator(messages=VALID_MESSAGES)
        evaluator._flow.assert_called_once()
        evaluator._multi_turn_flow.assert_not_called()
        assert "customer_satisfaction" in result


# region serialize_messages tests


@pytest.mark.unittest
class TestCustomerSatisfactionSerializeMessages:
    """Unit tests for the serialize_messages helper used by customer satisfaction."""

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
class TestCustomerSatisfactionNoneScoreHandling:
    """Regression tests for None-score handling in _parse_prompty_output.

    When the flow returns a skipped/None score, _parse_prompty_output must short-circuit
    on the skipped status and return score=None instead of crashing on ``None >= threshold``.
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow returns a skipped result."""
        evaluator = _create_mocked_evaluator()
        evaluator._flow = MagicMock(side_effect=create_none_score_flow_side_effect())
        result = evaluator(query="How do I return an item?", response="You can return within 30 days.")
        assert result["customer_satisfaction"] is None
        assert result["customer_satisfaction_result"] == "skipped"

    def test_conversation_level_none_score_does_not_crash(self):
        """Conversation-level eval with score=None from _multi_turn_flow returns a skipped result."""
        evaluator = _create_mocked_evaluator()
        evaluator._multi_turn_flow = MagicMock(
            side_effect=create_none_score_flow_side_effect(reason="No agent responses to evaluate.")
        )
        result = evaluator(messages=VALID_MESSAGES)
        assert result["customer_satisfaction"] is None
        assert result["customer_satisfaction_result"] == "skipped"


# endregion


@pytest.mark.unittest
class TestCustomerSatisfactionValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
    ConversationValidatorUnitTests,
    ConversationValidatorToolCheckUnitTests,
    ConversationSerializationUnitTests,
    AgentResponseReformatUnitTests,
):
    """Low-level unit tests for customer_satisfaction's repeated validators, utils and methods."""

    evaluator_class = CustomerSatisfactionEvaluator
