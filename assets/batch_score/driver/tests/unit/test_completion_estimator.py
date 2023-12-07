# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest

from src.batch_score.batch_pool.quota.estimators import CompletionEstimator


@pytest.mark.parametrize("request_obj, expected_cost", [
    ({"prompt": "Hello world!", "max_tokens": 10}, 3 + 10),
    ({"prompt": "<|endoftext|>", "max_tokens": 10}, 7 + 10),
    ({"prompt": "x"*10000, "max_tokens": 10}, 1250 + 10),
    ({"prompt": "x"*1000000, "max_tokens": 10}, 165776 + 10),
    ({"prompt": " x"*1000000, "max_tokens": 10}, 1000000 + 10),
])
def test_estimate_request_cost(request_obj, expected_cost):
    estimator = CompletionEstimator()

    assert estimator.estimate_request_cost(request_obj) == expected_cost


@pytest.mark.parametrize("request_obj, response_obj, expected_cost", [
    ({"prompt": "Hello world!", "max_tokens": 10}, {"usage": {"prompt_tokens": 123}}, 123 + 10),
])
def test_estimate_response_cost(request_obj, response_obj, expected_cost):
    estimator = CompletionEstimator()

    assert estimator.estimate_response_cost(request_obj, response_obj) == expected_cost


def test_invalid_get_prompt():
    estimator = CompletionEstimator()

    chat_completion_payload = {"messages": [{"role": "user", "content": "Hello!"}]}

    with pytest.raises(Exception):
        estimator._get_prompt(chat_completion_payload)
