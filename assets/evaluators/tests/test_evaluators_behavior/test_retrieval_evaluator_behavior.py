# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for Retrieval Evaluator — None score handling."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

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
class TestRetrievalJudgeOutputValidation:
    """Covers normalization and safe invalid-output diagnostics."""

    @pytest.mark.parametrize("score", [1, 5, "4", 4.0])
    def test_valid_scores_are_normalized(self, score):
        """Integer-equivalent numeric values produce valid results."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(
            return_value={
                "llm_output": {
                    "properties": {"thought_chain": "brief reasoning"},
                    "reason": "Relevant context.",
                    "score": score,
                    "status": "completed",
                }
            }
        )

        result = asyncio.run(evaluator._do_eval({"query": "question", "context": "context"}))

        assert result["retrieval_score"] == float(score)

    @pytest.mark.parametrize(
        ("llm_output", "expected_error"),
        [
            ({"reason": "No score", "status": "completed"}, "MissingScore"),
            ({"score": None, "status": "completed"}, "MissingScore"),
            ({"score": True, "status": "completed"}, "InvalidScoreType"),
            ({"score": "good", "status": "completed"}, "InvalidScoreType"),
            ({"score": 3.5, "status": "completed"}, "InvalidScoreType"),
            ({"score": float("nan"), "status": "completed"}, "NonFiniteScore"),
            ({"score": 6, "status": "completed"}, "ScoreOutOfRange"),
            ({"score": 4, "status": "unknown"}, "UnexpectedStatus"),
        ],
    )
    def test_invalid_scores_return_safe_diagnostic(self, llm_output, expected_error):
        """Invalid judge outputs identify the contract failure without exposing content."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(return_value={"llm_output": llm_output})

        with pytest.raises(EvaluationException, match=expected_error):
            asyncio.run(evaluator._do_eval({"query": "question", "context": "context"}))

        assert evaluator._flow.await_count == 1

    @pytest.mark.parametrize(
        ("prompty_output", "expected_error"),
        [
            ({}, "EmptyResponse"),
            ({"llm_output": ""}, "EmptyResponse"),
            ({"llm_output": "{}"}, "MissingScore"),
            ({"llm_output": "not-json"}, "MalformedJson"),
            ({"llm_output": []}, "UnexpectedResponseShape"),
        ],
    )
    def test_invalid_response_shapes_return_safe_diagnostic(self, prompty_output, expected_error):
        """Malformed response shapes produce a categorized system error."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(return_value=prompty_output)

        with pytest.raises(EvaluationException, match=expected_error):
            asyncio.run(evaluator._do_eval({"query": "question", "context": "context"}))

        assert evaluator._flow.await_count == 1

    def test_explicit_legacy_text_score_is_supported(self):
        """A clearly labeled legacy text score remains backward compatible."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(return_value={"llm_output": "score: 4"})

        result = asyncio.run(evaluator._do_eval({"query": "question", "context": "context"}))

        assert result["retrieval_score"] == 4

    def test_unrelated_digits_are_not_treated_as_legacy_score(self):
        """Dates and other incidental digits must not become evaluator scores."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(
            return_value={"llm_output": "Rated on 2026-10-01; quality could not be determined."}
        )

        with pytest.raises(EvaluationException, match="MalformedJson"):
            asyncio.run(evaluator._do_eval({"query": "question", "context": "context"}))

    def test_invalid_output_diagnostics_do_not_expose_raw_content(self, caplog):
        """Malformed-output diagnostics contain categories but not customer content."""
        canary = "SENSITIVE_RETRIEVAL_CONTENT_7f3d"
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        evaluator._flow = AsyncMock(
            return_value={"llm_output": canary, "finish_reason": "stop"}
        )

        with caplog.at_level(logging.WARNING):
            with pytest.raises(EvaluationException, match="MalformedJson") as error:
                asyncio.run(evaluator._do_eval({"query": canary, "context": canary}))

        assert canary not in caplog.text
        assert canary not in str(error.value)
        assert canary not in error.value.internal_message


@pytest.mark.unittest
class TestRetrievalConversationContextExtraction:
    """Covers retrieval context extraction from conversation tool outputs."""

    def test_query_response_extracts_tool_context(self):
        """Response-side tool outputs become context for an explicit query."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        response = [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_result": {"sourceData": {"snippet": "The warranty is 24 months."}},
                    }
                ],
            },
            {"role": "assistant", "content": "The warranty is 24 months."},
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(
            query="What is the warranty?",
            response=response,
        )

        assert inputs == [
            {
                "query": "What is the warranty?",
                "context": '{"sourceData": {"snippet": "The warranty is 24 months."}}',
            }
        ]

    def test_string_response_is_rejected(self):
        """Plain response text cannot be treated as retrieved context."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        query = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Latest question"},
        ]

        with pytest.raises(EvaluationException, match="'response' must be a list of messages"):
            evaluator._convert_kwargs_to_eval_input(
                query=query,
                response="Retrieved context",
            )

    def test_message_query_uses_latest_user_with_response_tool_context(self):
        """A message-list query contributes only its latest user text."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        query = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Latest question"},
        ]
        response = [{"role": "tool", "content": "Retrieved context"}]

        inputs = evaluator._convert_kwargs_to_eval_input(query=query, response=response)

        assert inputs == [{"query": "Latest question", "context": "Retrieved context"}]

    def test_response_without_tool_output_is_not_applicable(self):
        """A response with no tool messages cannot provide retrieval context."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        with pytest.raises(EvaluationException, match="No valid query or tool output"):
            evaluator._convert_kwargs_to_eval_input(
                query="Question",
                response=[{"role": "assistant", "content": "Answer"}],
            )

    def test_response_without_query_is_rejected(self):
        """A response without a query must not evaluate with an empty query."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        with pytest.raises(EvaluationException, match="'query' must be a string or a list of messages"):
            evaluator._convert_kwargs_to_eval_input(response=[{"role": "tool", "content": "Ctx"}])

    def test_query_without_user_text_is_not_applicable(self):
        """A message-list query without user text cannot produce a retrieval query."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        with pytest.raises(EvaluationException, match="No valid query or tool output"):
            evaluator._convert_kwargs_to_eval_input(
                query=[{"role": "assistant", "content": "Answer"}],
                response=[{"role": "tool", "content": "Ctx"}],
            )

    def test_non_string_non_list_query_is_rejected(self):
        """An unsupported query type raises a classified user error instead of a TypeError."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        with pytest.raises(EvaluationException, match="'query' must be a string or a list of messages"):
            evaluator._convert_kwargs_to_eval_input(
                query=5,
                response=[{"role": "tool", "content": "Ctx"}],
            )

    def test_query_context_is_preferred_over_response(self):
        """Explicit context keeps the legacy query/context behavior and ignores response."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        inputs = evaluator._convert_kwargs_to_eval_input(
            query="Question",
            context="Explicit context",
            response=[{"role": "tool", "content": "Response context"}],
        )

        assert inputs[0]["query"] == "Question"
        assert inputs[0]["context"] == "Explicit context"

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

    def test_explicit_context_uses_latest_user_message_as_query(self):
        """Messages with explicit context derive the query from the latest user turn."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "Latest question"},
            {"role": "assistant", "content": "Latest answer"},
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(messages=messages, context="Explicit context")

        assert inputs == [{"query": "Latest question", "context": "Explicit context"}]

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

    def test_conversation_object_uses_tool_context_extraction(self):
        """Conversation model objects expose messages through an attribute."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {"role": "user", "content": "What is the warranty?"},
            {"role": "tool", "content": "The warranty is 24 months."},
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(
            conversation=SimpleNamespace(messages=messages),
        )

        assert inputs == [
            {
                "query": "What is the warranty?",
                "context": "The warranty is 24 months.",
            }
        ]

    @pytest.mark.parametrize("messages", [None, [], "not-a-list"])
    def test_invalid_messages_are_rejected(self, messages):
        """Invalid conversation message collections produce a user error."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        if messages is None:
            with pytest.raises(EvaluationException):
                evaluator._convert_kwargs_to_eval_input(messages=messages)
        else:
            with pytest.raises(EvaluationException, match="non-empty list"):
                evaluator._convert_kwargs_to_eval_input(messages=messages)

    def test_explicit_query_overrides_single_derived_query(self):
        """An explicitly mapped query wins for a single derived retrieval turn."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {"role": "user", "content": "Derived query"},
            {"role": "tool", "content": "Retrieved context"},
        ]

        inputs = evaluator._convert_kwargs_to_eval_input(
            messages=messages,
            query="Explicit query",
        )

        assert inputs == [{"query": "Explicit query", "context": "Retrieved context"}]

    def test_context_helpers_handle_defensive_input_shapes(self):
        """Helper branches safely normalize malformed and scalar message content."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")

        assert evaluator._extract_retrieval_turns(
            [None, {"role": "user", "content": "Question"}, {"role": "tool", "content": 42}]
        ) == []
        assert evaluator._get_latest_user_query([{"role": "assistant", "content": "Answer"}]) == ""
        assert evaluator._extract_message_text(None) == ""
        assert evaluator._extract_message_text(
            ["ignored", {"type": "tool_call"}, {"type": "text", "text": "kept"}]
        ) == "kept"
        assert evaluator._extract_tool_message_context(None) == []
        assert evaluator._extract_tool_message_context(
            [
                "plain output",
                None,
                {"type": "tool_call"},
                {"output": 7},
            ]
        ) == ["plain output", "7"]
        assert evaluator._stringify_tool_output(" output ") == "output"
        assert evaluator._stringify_tool_output(None) == ""

    def test_messages_without_tool_output_are_not_applicable(self):
        """Conversation input without retrieval evidence returns a user-facing skip."""
        evaluator = create_mocked_evaluator(RetrievalEvaluator, "retrieval")
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
        ]

        with pytest.raises(EvaluationException, match="No valid context"):
            evaluator._convert_kwargs_to_eval_input(messages=messages)
