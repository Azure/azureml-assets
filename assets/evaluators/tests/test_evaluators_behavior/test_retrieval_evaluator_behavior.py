# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import asyncio
import json

import pytest
from azure.ai.evaluation._exceptions import EvaluationException

from .base_validator_unit_test import (
    CorePromptyValidatorUnitTests,
    MessagePreprocessUnitTests,
    SuperDoEvalNotApplicableUnitTests,
)
from ...builtin.retrieval.evaluator._retrieval import RetrievalEvaluator
from ..common.evaluator_mock_config import (
    INTERMEDIATE_FUNCTION_CALL_RESPONSE,
    create_mocked_evaluator,
    run_none_score_not_applicable,
)


# region None score handling tests

@pytest.mark.unittest
class TestRetrievalNoneScoreHandling:
    """Tests for None score handling in _do_eval (math.isnan fix).

    When _return_not_applicable_result returns score=None, _do_eval must not
    crash on math.isnan(None).
    """

    def test_turn_level_none_score_does_not_crash(self):
        """Turn-level eval with score=None from _flow should not raise TypeError."""
        run_none_score_not_applicable(
            RetrievalEvaluator,
            "retrieval",
            query="What are the office hours?",
            context="The office is open Monday through Friday from 9 AM to 5 PM.",
        )


# endregion


@pytest.mark.unittest
class TestRetrievalValidatorUnit(
    CorePromptyValidatorUnitTests,
    SuperDoEvalNotApplicableUnitTests,
    MessagePreprocessUnitTests,
):
    """Low-level unit tests for retrieval's repeated validators, utils and methods."""

    evaluator_class = RetrievalEvaluator


# region _do_eval override branch coverage

@pytest.mark.unittest
class TestRetrievalDoEvalBranches:
    """Cover retrieval's override ``_do_eval`` intermediate and list-preprocessing branches."""

    def test_intermediate_response_not_applicable(self):
        """An intermediate (function_call) response short-circuits to a not-applicable result."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        result = asyncio.run(evaluator._do_eval({"response": INTERMEDIATE_FUNCTION_CALL_RESPONSE}))
        assert result["retrieval_result"] == "not_applicable"

    def test_list_inputs_are_preprocessed(self):
        """List-typed query and response inputs are preprocessed before the flow call."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        result = asyncio.run(
            evaluator._do_eval(
                {
                    "query": [{"role": "user", "content": [{"type": "text", "text": "What are the hours?"}]}],
                    "response": [{"role": "assistant", "content": [{"type": "text", "text": "9 to 5."}]}],
                    "context": "The office is open 9 to 5.",
                }
            )
        )
        assert result["retrieval_score"] == 5


@pytest.mark.unittest
class TestRetrievalConversationContextExtraction:
    """Covers retrieval context extraction from conversation tool outputs."""

    def test_custom_knowledge_base_result_is_extracted_without_tool_filtering(self):
        """Nested Azure AI Search references from a custom tool remain intact."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        reference = {
            "type": "azureBlob",
            "sourceData": {
                "blob_url": "https://example.test/x-t5.md",
                "snippet": "The X-T5 warranty is 24 months.",
            },
            "rerankerScore": 3.990096,
        }
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the warranty period?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call-1",
                        "name": "knowledge_base_retrieve",
                        "arguments": {"query": "X-T5 warranty"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{"type": "tool_result", "tool_result": [reference]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "The warranty is 24 months."}],
            },
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(messages=messages)

        assert len(inputs) == 1
        assert inputs[0]["query"] == "What is the warranty period?"
        assert json.loads(inputs[0]["context"]) == [reference]

    def test_openapi_and_search_outputs_are_grouped_by_user_turn(self):
        """Specialized tool output types are extracted in chronological order."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "What is the capital?"}],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "openapi_call_output",
                        "output": {"country": "France", "capital": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is its population?"}],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "azure_ai_search_call_output",
                        "output": [{"content": "Paris has over two million residents."}],
                    }
                ],
            },
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(messages=messages)

        assert inputs == [
            {
                "query": "What is the capital?",
                "context": '{"capital": "Paris", "country": "France"}',
            },
            {
                "query": "What is its population?",
                "context": '[{"content": "Paris has over two million residents."}]',
            },
        ]

    def test_explicit_context_takes_precedence_over_tool_outputs(self):
        """Explicitly mapped context retains existing turn-level behavior."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the warranty?"}],
            },
            {
                "role": "tool",
                "content": [{"type": "tool_result", "tool_result": "Derived context"}],
            },
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(
            messages=messages,
            context="Explicit context",
        )

        assert inputs == [{"query": "What is the warranty?", "context": "Explicit context"}]

    def test_conversation_wrapper_uses_tool_context_extraction(self):
        """The documented conversation input follows the same messages path."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is the warranty?"}],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": {
                            "sourceData": {"snippet": "The warranty is 24 months."}
                        },
                    }
                ],
            },
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(
            conversation={"messages": messages},
        )

        assert inputs == [
            {
                "query": "What is the warranty?",
                "context": '{"sourceData": {"snippet": "The warranty is 24 months."}}',
            }
        ]

    def test_messages_without_tool_output_are_not_applicable(self):
        """Conversation input without retrieval evidence returns a user-facing skip."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]

        with pytest.raises(EvaluationException, match="No valid context"):
            evaluator._convert_kwargs_to_eval_input(messages=messages)
