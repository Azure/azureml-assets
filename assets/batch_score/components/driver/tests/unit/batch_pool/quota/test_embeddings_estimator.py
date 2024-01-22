# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for embeddings estimator."""

import pytest

from src.batch_score.batch_pool.quota.estimators import EmbeddingsEstimator


@pytest.mark.parametrize("input_prompt, expected_count", [
    ("one string", 2),
    (["a", "list", "of", "strings"], (1, 1, 1, 1))
])
def test_tiktoken_count(input_prompt, expected_count):
    estimator = EmbeddingsEstimator()
    actual_count = estimator.calc_tokens_with_tiktoken(input_prompt)
    assert actual_count == expected_count


@pytest.mark.parametrize("input_prompt, expected_cost", [
     ("Hello World", (2,)),
     (["Hello", "World"], (1, 1))
])
def test_estimate_request_cost(input_prompt, expected_cost):
    estimator = EmbeddingsEstimator()
    actual_cost = estimator.estimate_request_cost({"input": input_prompt})
    assert actual_cost == expected_cost


@pytest.mark.parametrize("input_prompt, expected_count", [
    ("Hello World", 2),
    (["Hello", "World"], 2)
])
def test_estimate_response_cost(input_prompt, expected_count):
    estimator = EmbeddingsEstimator()
    actual_count = estimator.estimate_response_cost(
        {"input": input_prompt},
        {"usage": {"total_tokens": 2}}
        )
    assert actual_count == expected_count
