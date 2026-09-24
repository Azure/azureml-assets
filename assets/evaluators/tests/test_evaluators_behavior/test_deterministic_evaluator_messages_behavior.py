# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Behavioral tests for deterministic evaluators accepting conversation messages."""

import json

import pytest

from ...builtin.bbeh.evaluator import _bbeh
from ...builtin.bbeh.evaluator._bbeh import BBEHEvaluator
from ...builtin.bleu_score.evaluator import _bleu
from ...builtin.bleu_score.evaluator._bleu import BleuScoreEvaluator
from ...builtin.f1_score.evaluator import _f1_score
from ...builtin.f1_score.evaluator._f1_score import F1ScoreEvaluator
from ...builtin.gleu_score.evaluator import _gleu
from ...builtin.gleu_score.evaluator._gleu import GleuScoreEvaluator
from ...builtin.ifeval.evaluator import _ifeval
from ...builtin.ifeval.evaluator._ifeval import IFEvalEvaluator
from ...builtin.meteor_score.evaluator import _meteor
from ...builtin.meteor_score.evaluator._meteor import MeteorScoreEvaluator
from ...builtin.regex_match.evaluator import _regex_match
from ...builtin.regex_match.evaluator._regex_match import RegexMatchEvaluator
from ...builtin.rouge_score.evaluator import _rouge
from ...builtin.rouge_score.evaluator._rouge import RougeScoreEvaluator


PARSER_MODULES = [_bbeh, _bleu, _f1_score, _gleu, _ifeval, _meteor, _regex_match, _rouge]

MESSAGES = [
    {"role": "user", "content": "Give an unrelated answer."},
    {"role": "assistant", "content": "Completely unrelated wording."},
    {"role": "user", "content": "Give the final answer."},
    {"role": "assistant", "content": [{"type": "tool_call", "name": "lookup", "arguments": {}}]},
    {"role": "tool", "content": [{"type": "tool_result", "tool_result": "irrelevant"}]},
    {"role": "assistant", "content": "The quick brown fox"},
]


@pytest.mark.unittest
@pytest.mark.parametrize("module", PARSER_MODULES)
@pytest.mark.parametrize(
    "response,expected",
    [
        ("final answer", "final answer"),
        (json.dumps([{"role": "assistant", "content": "final answer"}]), "final answer"),
        (
            json.dumps(
                [{"role": "assistant", "content": [{"type": "text", "text": "final answer"}]}]
            ),
            "final answer",
        ),
        ([{"role": "assistant", "content": "final answer"}], "final answer"),
        (
            [{"role": "assistant", "content": [{"type": "text", "text": "final answer"}]}],
            "final answer",
        ),
        (
            [
                {"role": "assistant", "content": "intermediate text"},
                {"role": "assistant", "content": "final answer"},
            ],
            "final answer",
        ),
        (
            [
                {"role": "assistant", "content": "final answer"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_call", "name": "lookup", "arguments": {}}],
                },
            ],
            "final answer",
        ),
        ([{"role": "tool", "content": [{"type": "tool_result", "tool_result": "x"}]}], ""),
        ("", ""),
        ("[{bad json", "[{bad json"),
        ('{"answer": "final answer"}', '{"answer": "final answer"}'),
    ],
)
def test_response_parser_representations(module, response, expected):
    """Response parsing handles text, JSON messages, direct lists, and fallback inputs."""
    assert module._parse_response_for_evaluation(response) == expected


@pytest.mark.unittest
@pytest.mark.parametrize("module", PARSER_MODULES)
@pytest.mark.parametrize(
    "messages",
    [
        [123],
        [{"role": "narrator", "content": "hello"}],
        [{"role": "user", "content": "hello"}],
        [
            {"role": "user", "content": "Give the final answer."},
            {"role": "assistant", "content": [{"type": "tool_call", "name": "lookup", "arguments": {}}]},
            {"role": "tool", "content": [{"type": "tool_result", "tool_result": "irrelevant"}]},
        ],
    ],
)
def test_messages_with_no_final_text_returns_empty_string(module, messages):
    """A conversation with no assistant text is treated like an empty text response, not an error."""
    assert module._response_from_messages(messages) == ""


@pytest.mark.unittest
@pytest.mark.parametrize("module", PARSER_MODULES)
@pytest.mark.parametrize("messages", ["not a list", []])
def test_messages_reject_invalid_input(module, messages):
    """Only a structurally invalid (non-list or empty) messages value raises."""
    with pytest.raises(ValueError):
        module._response_from_messages(messages)


@pytest.mark.unittest
@pytest.mark.parametrize("module", PARSER_MODULES)
def test_messages_without_user_turn_still_extracts_assistant_text(module):
    """No user turn is required; the latest assistant text is still extracted."""
    assert module._response_from_messages([{"role": "assistant", "content": "hello"}]) == "hello"


@pytest.mark.unittest
@pytest.mark.parametrize("module", PARSER_MODULES)
def test_messages_search_stops_at_latest_user_boundary(module):
    """Text from an earlier turn is never used when the latest turn has no text."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "tool_call", "name": "lookup", "arguments": {}}]},
    ]
    assert module._response_from_messages(messages) == ""


@pytest.mark.unittest
@pytest.mark.parametrize(
    "evaluator,result_key,minimum_score",
    [
        (BBEHEvaluator(), "bbeh", 1.0),
        (BleuScoreEvaluator(), "bleu", 0.9),
        (F1ScoreEvaluator(), "f1_score", 0.9),
        (GleuScoreEvaluator(), "gleu", 0.9),
        (MeteorScoreEvaluator(), "meteor", 0.9),
        (RougeScoreEvaluator(rouge_type="rouge1"), "rouge", 0.9),
    ],
)
def test_messages_use_only_final_agent_turn(evaluator, result_key, minimum_score):
    """Earlier assistant text must not affect deterministic response scoring."""
    result = evaluator(messages=MESSAGES, ground_truth="The quick brown fox")

    assert result[result_key] >= minimum_score


@pytest.mark.unittest
def test_ifeval_messages_use_only_final_agent_turn():
    """Applies IFEval instructions to the final agent response."""
    result = IFEvalEvaluator()(
        messages=MESSAGES,
        instruction_id_list='["punctuation:no_comma"]',
        instruction_kwargs='[{}]',
    )

    assert result["ifeval_strict"] is True


@pytest.mark.unittest
def test_regex_match_messages_use_only_final_agent_turn():
    """Regex matching evaluates the final agent response."""
    result = RegexMatchEvaluator(patterns=[r"^The quick brown fox$"])(messages=MESSAGES)

    assert result["regex_match"] is True


@pytest.mark.unittest
@pytest.mark.parametrize(
    "evaluator,result_key",
    [
        (BleuScoreEvaluator(), "bleu"),
        (F1ScoreEvaluator(), "f1_score"),
        (GleuScoreEvaluator(), "gleu"),
        (MeteorScoreEvaluator(), "meteor"),
        (RougeScoreEvaluator(rouge_type="rouge1"), "rouge"),
    ],
)
def test_array_response_without_text_scores_like_empty_string_response(evaluator, result_key):
    """An array response with no assistant text scores the same as an empty string response."""
    array_result = evaluator(
        response=[{"role": "tool", "content": [{"type": "tool_result", "tool_result": "x"}]}],
        ground_truth="The quick brown fox",
    )
    empty_result = evaluator(response="", ground_truth="The quick brown fox")

    assert array_result[result_key] == empty_result[result_key]
