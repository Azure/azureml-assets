# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import random
import string
import tempfile

import mltable
import pandas as pd
import pytest

from src.batch_score.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus,
)
from src.batch_score.utils import common

MLTable_yaml = """
type: mltable

paths:
  - pattern: data.txt

transformations:
  - read_json_lines:
      encoding: utf8
      include_path_column: false
"""

VALID_DATAFRAMES = [
    [[{"arrayCol": [1, 2]}, {"arrayCol": [3, 4]}], ['{"arrayCol": [1, 2]}', '{"arrayCol": [3, 4]}']],
    [[{"max_tokens": 1}, {"max_tokens": None}, {"max_tokens": 2}], ['{"max_tokens": 1}', '{}', '{"max_tokens": 2}']],
    [[{"stop": ["<|im_end|>", "\n", "\\n", "```"]}], ['{"stop": ["<|im_end|>", "\\n", "\\\\n", "```"]}']],
]


@pytest.mark.parametrize("obj_list, expected_result", VALID_DATAFRAMES)
def test_convert_to_list_happy_path(obj_list: "list[any]", expected_result: "list[str]"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(file=os.path.join(tmp_dir, "data.txt"), mode="w+") as text_file:
            for obj in obj_list:
                str_json = json.dumps(obj)
                text_file.write(str_json + "\n")

        with open(file=os.path.join(tmp_dir, "MLTable"), mode="w+") as mltable_file:
            mltable_file.write(MLTable_yaml)

        _mltable = mltable.load(tmp_dir)
        df_data = _mltable.to_pandas_dataframe()

        result = common.convert_to_list(df_data)
        assert result == expected_result


@pytest.mark.parametrize("additional_properties", [
    (None),
    ({"max_tokens": 11})
])
def test_convert_to_list_batch_size_20(additional_properties):
    batch_size_per_request = 20
    num_batches = 4
    full_batches = num_batches - 1
    additional_rows = 2
    obj_list = [{"input": __random_string()} for i in range(20 * full_batches + additional_rows)]
    input_df = pd.DataFrame(obj_list)
    expected_first_batch = {"input": input_df.input[:batch_size_per_request].values.tolist()}
    expected_last_batch = {"input": input_df.input[full_batches * batch_size_per_request:].values.tolist()}

    if additional_properties is not None:
        for k, v in additional_properties.items():
            expected_first_batch[k] = v
            expected_last_batch[k] = v

    expected_first_batch = json.dumps(expected_first_batch)
    expected_last_batch = json.dumps(expected_last_batch)

    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(file=os.path.join(tmp_dir, "data.txt"), mode="w+") as text_file:
            for obj in obj_list:
                str_json = json.dumps(obj)
                text_file.write(str_json + "\n")

        with open(file=os.path.join(tmp_dir, "MLTable"), mode="w+") as mltable_file:
            mltable_file.write(MLTable_yaml)

        _mltable = mltable.load(tmp_dir)
        df_data = _mltable.to_pandas_dataframe()

        additional_properties_string = json.dumps(additional_properties)
        result = common.convert_to_list(df_data, additional_properties_string, batch_size_per_request)
        assert len(result) == num_batches
        assert result[0] == expected_first_batch
        assert result[-1] == expected_last_batch


@pytest.mark.parametrize("tiktoken_failed",
                         [True, False])
def test_convert_result_list_batch_size_one(tiktoken_failed):
    # Arrange
    batch_size_per_request = 1
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)[0]
    outputlist = __get_output_data(batch_size_per_request)
    result = __get_scoring_result_for_batch(batch_size_per_request,
                                            inputstring,
                                            outputlist,
                                            tiktoken_failed=tiktoken_failed)
    result_list.append(result)

    # Act
    actual = common.convert_result_list(result_list, batch_size_per_request)
    actual_obj = json.loads(actual[0])

    # Assert
    assert len(actual) == 1
    assert actual_obj["status"] == "SUCCESS"
    assert "start" in actual_obj
    assert "end" in actual_obj
    assert actual_obj["request"]["input"] == inputstring
    assert actual_obj["response"]["usage"]["prompt_tokens"] == 1
    assert actual_obj["response"]["usage"]["total_tokens"] == 1


@pytest.mark.parametrize("tiktoken_failed",
                         [True, False])
def test_convert_result_list_failed_result(tiktoken_failed):
    # Arrange
    batch_size_per_request = 1
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)[0]
    result = __get_failed_scoring_result_for_batch(inputstring, tiktoken_failed=tiktoken_failed)
    result_list.append(result)

    # Act
    actual = common.convert_result_list(result_list, batch_size_per_request)
    actual_obj = json.loads(actual[0])

    # Assert
    assert len(actual) == 1
    assert actual_obj["status"] == "FAILURE"
    assert "start" in actual_obj
    assert "end" in actual_obj
    assert actual_obj["request"]["input"] == inputstring
    assert actual_obj["response"]["error"]["type"] == "invalid_request_error"
    assert "maximum context length is 8190 tokens" in actual_obj["response"]["error"]["message"]


@pytest.mark.parametrize("tiktoken_failed",
                         [True, False])
def test_convert_result_list_failed_result_batch(tiktoken_failed):
    # Arrange
    batch_size_per_request = 2

    inputlist = __get_input_batch(batch_size_per_request)
    result_list = [__get_failed_scoring_result_for_batch(inputlist, tiktoken_failed)]

    # Act
    actual = common.convert_result_list(result_list, batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request
    for idx, result in enumerate(actual):
        output_obj = json.loads(result)
        assert output_obj["request"]["input"] == inputlist[idx]

        assert output_obj["response"]["error"]["type"] == "invalid_request_error"
        assert "maximum context length is 8190 tokens" in output_obj["response"]["error"]["message"]


@pytest.mark.parametrize("reorder_results, online_endpoint_url, tiktoken_fails",
                         [(True, True, True),
                          (True, True, False),
                          (True, False, True),
                          (True, False, False),
                          (False, True, True),
                          (False, True, False),
                          (False, False, True),
                          (False, False, False)])
def test_convert_result_list_batch_20(
        monkeypatch,
        mock_get_logger,
        reorder_results,
        online_endpoint_url,
        tiktoken_fails):
    # Arrange
    batch_size_per_request = 20
    num_batches = 4
    full_batches = num_batches - 1
    additional_rows = 2

    result_list = []
    inputlists = []
    for n in range(full_batches):
        inputlist = __get_input_batch(batch_size_per_request)
        outputlist = __get_output_data(batch_size_per_request)
        result = __get_scoring_result_for_batch(
            batch_size_per_request,
            inputlist,
            outputlist,
            reorder_results,
            online_endpoint_url or tiktoken_fails)
        inputlists.extend(inputlist)
        result_list.append(result)
    inputlist = __get_input_batch(additional_rows)
    outputlist = __get_output_data(additional_rows)

    inputlists.extend(inputlist)
    result_list.append(__get_scoring_result_for_batch(
        additional_rows,
        inputlist,
        outputlist,
        reorder_results,
        online_endpoint_url or tiktoken_fails))

    # Mock what happens if estimates are not filled in when converting to output list
    if tiktoken_fails:
        __mock_tiktoken_permanent_failure(monkeypatch)
    else:
        __mock_tiktoken_estimate(monkeypatch)

    # Act
    actual = common.convert_result_list(result_list, batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request * full_batches + additional_rows
    for idx, result in enumerate(actual):
        output_obj = json.loads(result)
        assert output_obj["request"]["input"] == inputlists[idx]

        # Assign valid_batch_len for this result. This is the expected total_tokens.
        if idx >= batch_size_per_request * full_batches:
            # Final batch only has `additional_rows` length.
            valid_batch_len = additional_rows
        else:
            # All others have length equal to batch_size.
            valid_batch_len = batch_size_per_request

        # Assign valid_batch_idx for this result. This is used for validating prompt_tokens when tiktoken succeeds.
        # Index values in `response.data` are from [0, batch_size_per_request -1]
        valid_batch_idx = idx % batch_size_per_request

        assert output_obj["response"]["data"][0]["index"] == valid_batch_idx
        assert output_obj["response"]["usage"]["total_tokens"] == valid_batch_len
        if tiktoken_fails:
            # Prompt tokens will equal total tokens (equals batch length)
            assert output_obj["response"]["usage"]["prompt_tokens"] == valid_batch_len
        elif online_endpoint_url:
            # Prompt tokens is batch index (see helper function: `__mock_tiktoken_estimate`)
            assert output_obj["response"]["usage"]["prompt_tokens"] == valid_batch_idx
        else:
            # Batch pool case; prompt tokens is 10 + batch index (see helper function: `__get_token_counts`)
            assert output_obj["response"]["usage"]["prompt_tokens"] == valid_batch_idx + 10


def test_incorrect_data_length_raises():
    # Arrange
    batch_size_per_request = 2
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)
    outputlist = __get_output_data(0)
    result = __get_scoring_result_for_batch(batch_size_per_request, inputstring, outputlist)
    result_list.append(result)

    # Act
    with pytest.raises(Exception) as excinfo:
        common.convert_result_list(result_list, batch_size_per_request)

    # Assert
    assert "Result data length 0 != 2 request batch length." in str(excinfo.value)


def test_endpoint_response_is_not_json(mock_get_logger):
    # Arrange failed response payload as a string
    batch_size_per_request = 10
    inputs = __get_input_batch(batch_size_per_request)

    result = ScoringResult(
        status=ScoringResultStatus.FAILURE,
        start=0,
        end=0,
        request_metadata="Not important",
        response_headers="Headers",
        num_retries=2,
        token_counts=(1,) * batch_size_per_request,
        request_obj={"input": inputs},
        response_body=json.dumps({"object": "list",
                                  "error": {
                                      "message": "This model's maximum context length is 8190 tokens, "
                                                 "however you requested 10001 tokens "
                                                 "(10001 in your prompt; 0 for the completion). "
                                                 "Please reduce your prompt; or completion length.",
                                      "type": "invalid_request_error",
                                      "param": None,
                                      "code": None
                                    }}))
    # Act
    actual = common.convert_result_list([result], batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request
    for idx, result in enumerate(actual):
        unit = json.loads(result)
        assert unit['request']['input'] == inputs[idx]
        assert type(unit['response']) is str
        assert 'maximum context length' in unit['response']


def __get_scoring_result_for_batch(batch_size, inputlist, outputlist, reorder_results=False, tiktoken_failed=False):
    token_counts = __get_token_counts(tiktoken_failed, inputlist)
    result = ScoringResult(
        status=ScoringResultStatus.SUCCESS,
        start=0,
        end=0,
        request_metadata="Not important",
        response_headers="Headers",
        num_retries=2,
        token_counts=token_counts,
        request_obj={"input": inputlist},
        response_body={"object": "list",
                       "data": outputlist,
                       "model": "text-embedding-ada-002",
                       "usage": {
                                   "prompt_tokens": batch_size,
                                   "total_tokens": batch_size
                       }})
    if reorder_results:
        random.shuffle(result.response_body["data"])
    return result


def __get_failed_scoring_result_for_batch(inputlist, tiktoken_failed=False):
    token_counts = __get_token_counts(tiktoken_failed, inputlist)
    result = ScoringResult(
        status=ScoringResultStatus.FAILURE,
        start=0,
        end=0,
        request_metadata="Not important",
        response_headers="Headers",
        num_retries=2,
        token_counts=token_counts,
        request_obj={"input": inputlist},
        response_body={"object": "list",
                       "error": {
                           "message": "This model's maximum context length is 8190 tokens, "
                                      "however you requested 10001 tokens "
                                      "(10001 in your prompt; 0 for the completion). "
                                      "Please reduce your prompt; or completion length.",
                           "type": "invalid_request_error",
                           "param": None,
                           "code": None
                       }})
    return result


def __get_input_batch(batch_size):
    return [__random_string() for i in range(batch_size)]


def __get_output_data(batch_size):
    return [__embedding_output_info(i) for i in range(batch_size)]


def __get_token_counts(tiktoken_failed, inputlist):
    if tiktoken_failed:
        return []
    else:
        if isinstance(inputlist, str):
            inputlist = [inputlist]
        return [i + 10 for i, j in enumerate(inputlist)]


def __embedding_output_info(idx):
    return {
        "object": "embedding",
        "embedding": [random.random() for j in range(1)],
        "index": idx
    }


def __random_string(length: int = 10):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(length))


def __mock_tiktoken_permanent_failure(monkeypatch):
    def mock_tiktoken_failure(*args):
        return 1
    monkeypatch.setattr(common.embeddings.EmbeddingsEstimator, "estimate_request_cost", mock_tiktoken_failure)


def __mock_tiktoken_estimate(monkeypatch):
    def mock_tiktoken_override(estimator, request_obj):
        return [i for i in range(len(request_obj['input']))]
    monkeypatch.setattr(common.embeddings.EmbeddingsEstimator, "estimate_request_cost", mock_tiktoken_override)
