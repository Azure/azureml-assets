# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


"""This file contains unit tests for the decoding logic."""

import pytest
import json
from generation_safety_quality.annotation_compute_histogram.run import (
    _parse_responses, _Job, _PromptData)


def create_response(n_tasks):
    """Create a response with n_tasks."""
    tasks = []
    for task_idx in range(n_tasks):
        # prefix can be 0, 1, or 2 '#'
        task_prefix = '#' * (task_idx % 3)
        # if prefix contains # then add a space
        task_prefix += " " if task_prefix else ""
        # suffix can be 0 or 1 '#' followed by colon and newline
        task_suffix = '#' * (task_idx % 2)
        task_suffix += str(task_idx) + ":\n"
        # rating is 1-5
        rating_dict = {'rating': task_idx % 5 + 1}
        tasks.append(
            task_prefix +
            "Task " +
            task_suffix +
            str(json.dumps(rating_dict)))
    return '\n\n'.join(tasks)


@pytest.mark.unit
class TestDecoding:
    """Test decoding logic."""

    @pytest.mark.parametrize(
        "n_tasks", list(range(1, 100)))
    def test_parse_responses_successful(self, n_tasks):
        """Test for successfully parsing responses."""
        job = _Job(
            job_idx=0,
            prompt_data=_PromptData(
                input_idx=list(range(n_tasks)),
                input_examples=[],  # not used
                prompt="prompt does not matter for this test",
                n_tokens_estimate=1000),
            request_params={},
            response_data={
                "finish_reason": "stop",
                "samples": [
                    create_response(n_tasks=n_tasks)
                ]
            })

        _parse_responses(job, 1)
        assert len(job.response_data["output_examples"][0]) == n_tasks
        assert [
            result["rating"] in list(range(1, 5))
            for result in job.response_data["output_examples"][0]]

    def test_parse_responses_failed(self):
        """Test for failing to parse responses."""
        job = _Job(
            job_idx=0,
            prompt_data=_PromptData(
                input_idx=list(range(5)),
                input_examples=[],  # not used
                prompt="prompt does not matter for this test",
                n_tokens_estimate=1000),
            request_params={},
            response_data={
                "finish_reason": "stop",
                "samples": [
                    "Arbitrary unexpected start of the response" +
                    create_response(n_tasks=5)
                ]
            })

        with pytest.raises(Exception) as e:
            _parse_responses(job, 1)
            assert "Failed splitting output into examples:" in str(e)
