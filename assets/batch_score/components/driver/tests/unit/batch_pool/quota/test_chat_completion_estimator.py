# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests chat completion estimator."""

import pytest

from src.batch_score.batch_pool.quota.estimators import ChatCompletionEstimator


@pytest.mark.parametrize("request_obj, expected_cost", [
    ({"messages": [{"role": "user", "content": "Hello world!"},
                   {"role": "user", "content": "Hello world!"}],
      "max_tokens": 10},
     3 + 3 + 10),
])
def test_estimate_request_cost(request_obj, expected_cost):
    """Test estimate request cost."""
    estimator = ChatCompletionEstimator()

    assert estimator.estimate_request_cost(request_obj) == expected_cost


@pytest.mark.parametrize("request_obj, response_obj, expected_cost", [
    ({"messages": [{"role": "user", "content": "Hello world!"}], "max_tokens": 10},
     {"usage": {"prompt_tokens": 123}},
     123 + 10),
])
def test_estimate_response_cost(request_obj, response_obj, expected_cost):
    """Test estimate response cost."""
    estimator = ChatCompletionEstimator()

    assert estimator.estimate_response_cost(request_obj, response_obj) == expected_cost


def test_valid__get_prompt():
    """Test get prompt valid case."""
    estimator = ChatCompletionEstimator()

    chat_completion_payload = {"messages": [{"role": "user", "content": "Hello "},
                                            {"role": "user", "content": "World!"}]}

    assert estimator._get_prompt(chat_completion_payload) == "Hello World!"


def test_invalid_get_prompt():
    """Test get prompt invalid case."""
    estimator = ChatCompletionEstimator()

    completion_payload = {"prompt": "Hello world!", "max_tokens": 10}

    with pytest.raises(Exception):
        estimator._get_prompt(completion_payload)
