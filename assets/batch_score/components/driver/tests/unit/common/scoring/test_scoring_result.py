# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for scoring result."""

from typing import Any, Optional
import pytest

from src.batch_score.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus,
)


def test_copy():
    """Test copy."""
    result = ScoringResult(
        ScoringResultStatus.SUCCESS,
        0,
        1,
        200,
        "request_obj",
        {},
        response_body={"usage": {"prompt_tokens": 2,
                                 "completion_tokens": 4,
                                 "total_tokens": 6}},
        response_headers=None,
        num_retries=0,
        omit=False,
        token_counts=(1, 2, 3))

    result2 = result.copy()
    result2.response_body["usage"]["prompt_tokens"] = 7
    result2.response_body["usage"]["total_tokens"] = 16
    result2.response_body["usage"]["completion_tokens"] = 9

    assert result.prompt_tokens == 2
    assert result.completion_tokens == 4
    assert result.total_tokens == 6
    assert result.response_body["usage"]["prompt_tokens"] == 2
    assert result.response_body["usage"]["completion_tokens"] == 4
    assert result.response_body["usage"]["total_tokens"] == 6

    assert result2.prompt_tokens is None
    assert result2.completion_tokens is None
    assert result2.total_tokens is None
    assert result2.response_body["usage"]["prompt_tokens"] == 7
    assert result2.response_body["usage"]["completion_tokens"] == 9
    assert result2.response_body["usage"]["total_tokens"] == 16

    assert result.status == result2.status
    assert result.model_response_code == result2.model_response_code
    assert result2.estimated_token_counts == (1, 2, 3)


@pytest.mark.parametrize(
    "response_usage, expected_prompt_tokens, expected_total_tokens, expected_completion_tokens",
    [
        ({
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 4,
                "total_tokens": 6
            }
        }, 2, 6, 4),
        ({}, None, None, None),
        ([], None, None, None),
        ({
            "usage": "Some unknown usage"
        }, None, None, None)
    ]
)
def test_usage_statistics(
    response_usage: Any,
    expected_prompt_tokens: Optional[int],
    expected_total_tokens: Optional[int],
    expected_completion_tokens: Optional[int],
):
    """Test usage statistics."""
    result = ScoringResult(
        ScoringResultStatus.SUCCESS,
        0,
        1,
        200,
        "request_obj",
        {},
        response_body=response_usage,
        response_headers=None,
        num_retries=0,
        omit=False,
        token_counts=(1, 2, 3))
    assert result.prompt_tokens == expected_prompt_tokens
    assert result.completion_tokens == expected_completion_tokens
    assert result.total_tokens == expected_total_tokens
