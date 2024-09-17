# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for common utils."""

import json
import os
import random
import string
import tempfile

import mltable
import pandas as pd
import pytest

from src.batch_score_oss.root.common.scoring.scoring_result import (
    ScoringResult,
    ScoringResultStatus,
)
from src.batch_score_oss.root.utils.v1_output_formatter import V1OutputFormatter
from src.batch_score_oss.root.utils.v2_output_formatter import V2OutputFormatter
from src.batch_score_oss.root.utils.v1_input_schema_handler import V1InputSchemaHandler
from src.batch_score_oss.root.utils.v2_input_schema_handler import V2InputSchemaHandler

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

NEW_SCHEMA_VALID_DATAFRAMES = [
    [
        [
            {"custom_id": "task_123", "method": "POST", "url": "/v1/completions",
             "body": {"model": "chat-sahara-4", "max_tokens": 1}},
            {"custom_id": "task_789", "method": "POST", "url": "/v1/completions",
             "body": {"model": "chat-sahara-4", "max_tokens": 2}}
        ],
        [
            '{"max_tokens": 1, "custom_id": "task_123"}', '{"max_tokens": 2, "custom_id": "task_789"}'
        ]
    ],
    [
        [
            {
                "custom_id": "task_123",
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": "chat-sahara-4",
                    "temperature": 0,
                    "max_tokens": 1024,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "prompt": "# You will be given a conversation between a chatbot called Sydney and Human..."
                }
            },
            {
                "custom_id": "task_456",
                "method": "POST",
                "url": "/v1/completions",
                "body": {
                    "model": "chat-sahara-4",
                    "temperature": 0,
                    "max_tokens": 1024,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "prompt": "# You will be given a conversation between a chatbot called Sydney and Human..."
                }
            }
        ],
        [
            ('{"temperature": 0, "max_tokens": 1024, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0,'
             ' "prompt": "# You will be given a conversation between a chatbot called Sydney and Human...",'
             ' "custom_id": "task_123"}'),
            ('{"temperature": 0, "max_tokens": 1024, "top_p": 1, "frequency_penalty": 0, "presence_penalty": 0,'
             ' "prompt": "# You will be given a conversation between a chatbot called Sydney and Human...",'
             ' "custom_id": "task_456"}')
        ]
    ]
]


@pytest.mark.parametrize("obj_list, expected_result", VALID_DATAFRAMES)
def test_convert_input_to_requests_happy_path(obj_list: "list[any]", expected_result: "list[str]"):
    """Test convert to list happy path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(file=os.path.join(tmp_dir, "data.txt"), mode="w+") as text_file:
            for obj in obj_list:
                str_json = json.dumps(obj)
                text_file.write(str_json + "\n")

        with open(file=os.path.join(tmp_dir, "MLTable"), mode="w+") as mltable_file:
            mltable_file.write(MLTable_yaml)

        _mltable = mltable.load(tmp_dir)
        df_data = _mltable.to_pandas_dataframe()

        input_handler = V1InputSchemaHandler()
        result = input_handler.convert_input_to_requests(df_data)
        assert result == expected_result


@pytest.mark.parametrize("additional_properties", [
    (None),
    ({"max_tokens": 11})
])
def test_convert_input_to_requests_batch_size_20(additional_properties):
    """Test convert to list batch size 20 case."""
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
        input_handler = V1InputSchemaHandler()
        result = input_handler.convert_input_to_requests(df_data, additional_properties_string, batch_size_per_request)
        assert len(result) == num_batches
        assert result[0] == expected_first_batch
        assert result[-1] == expected_last_batch


@pytest.mark.parametrize("obj_list, expected_result", NEW_SCHEMA_VALID_DATAFRAMES)
def test_new_schema_convert_input_to_requests_happy_path(obj_list: "list[any]", expected_result: "list[str]"):
    """Test convert to list happy path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(file=os.path.join(tmp_dir, "data.txt"), mode="w+") as text_file:
            for obj in obj_list:
                str_json = json.dumps(obj)
                text_file.write(str_json + "\n")

        with open(file=os.path.join(tmp_dir, "MLTable"), mode="w+") as mltable_file:
            mltable_file.write(MLTable_yaml)

        _mltable = mltable.load(tmp_dir)
        df_data = _mltable.to_pandas_dataframe()

        input_handler = V2InputSchemaHandler()
        result = input_handler.convert_input_to_requests(df_data)
        assert result == expected_result


@pytest.mark.parametrize("input_schema_version", [1, 2])
@pytest.mark.parametrize("tiktoken_failed", [True, False])
def test_output_formatter_batch_size_one(input_schema_version, tiktoken_failed):
    """Test convert result list batch size one case."""
    # Arrange
    batch_size_per_request = 1
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)[0]
    request_obj = {"input": inputstring, "custom_id": "task_123"}
    outputlist = __get_output_data(batch_size_per_request)
    result = __get_scoring_result_for_batch(batch_size_per_request,
                                            request_obj,
                                            outputlist,
                                            tiktoken_failed=tiktoken_failed)
    result_list.append(result)

    # Act
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output(result_list, batch_size_per_request)
    actual_obj = json.loads(actual[0])

    # Assert
    assert len(actual) == 1
    if input_schema_version == 1:
        assert actual_obj["status"] == "SUCCESS"
        assert "start" in actual_obj
        assert "end" in actual_obj
        assert actual_obj["request"]["input"] == inputstring
        assert actual_obj["response"]["usage"]["prompt_tokens"] == 1
        assert actual_obj["response"]["usage"]["total_tokens"] == 1
    elif input_schema_version == 2:
        assert actual_obj["response"]["status_code"] == 200
        assert actual_obj["response"]["body"]["usage"]["prompt_tokens"] == 1
        assert actual_obj["response"]["body"]["usage"]["total_tokens"] == 1


@pytest.mark.parametrize("input_schema_version", [1, 2])
@pytest.mark.parametrize("tiktoken_failed", [True, False])
def test_output_formatter_failed_result(input_schema_version, tiktoken_failed):
    """Test convert result list failed result case."""
    # Arrange
    batch_size_per_request = 1
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)[0]
    request_obj = {"input": inputstring, "custom_id": "task_123"}
    result = __get_failed_scoring_result_for_batch(request_obj, tiktoken_failed=tiktoken_failed)
    result_list.append(result)

    # Act
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output(result_list, batch_size_per_request)
    actual_obj = json.loads(actual[0])

    # Assert
    assert len(actual) == 1
    if input_schema_version == 1:
        assert actual_obj["status"] == "FAILURE"
        assert "start" in actual_obj
        assert "end" in actual_obj
        assert actual_obj["request"]["input"] == inputstring
        assert actual_obj["response"]["error"]["type"] == "invalid_request_error"
        assert "maximum context length is 8190 tokens" in actual_obj["response"]["error"]["message"]
    elif input_schema_version == 2:
        assert actual_obj["response"]["request_id"] is not None
        assert actual_obj["response"]["status_code"] == 400
        assert actual_obj["error"]["message"]["error"]["type"] == "invalid_request_error"
        assert "maximum context length is 8190 tokens" in actual_obj["error"]["message"]["error"]["message"]


@pytest.mark.parametrize("input_schema_version", [1, 2])
@pytest.mark.parametrize("tiktoken_failed", [True, False])
def test_output_formatter_failed_result_batch(input_schema_version, tiktoken_failed):
    """Test convert result list failed result batch case."""
    # Arrange
    batch_size_per_request = 2

    inputlist = __get_input_batch(batch_size_per_request)
    request_obj = {"input": inputlist, "custom_id": "task_123"}
    result_list = [__get_failed_scoring_result_for_batch(request_obj, tiktoken_failed)]

    # Act
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output(result_list, batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request
    for idx, result in enumerate(actual):
        output_obj = json.loads(result)
        if input_schema_version == 1:
            assert output_obj["request"]["input"] == inputlist[idx]
            assert output_obj["response"]["error"]["type"] == "invalid_request_error"
            assert "maximum context length is 8190 tokens" in output_obj["response"]["error"]["message"]
        elif input_schema_version == 2:
            assert output_obj["response"]["request_id"] is not None
            assert output_obj["response"]["status_code"] == 400
            assert output_obj["error"]["message"]["error"]["type"] == "invalid_request_error"
            assert "maximum context length is 8190 tokens" in output_obj["error"]["message"]["error"]["message"]


def test_output_formatter_failed_empty_response_headers_batch():
    """Test convert result list failed result batch case."""
    # Arrange
    batch_size_per_request = 2

    inputlist = __get_input_batch(batch_size_per_request)
    request_obj = {"input": inputlist, "custom_id": "task_123"}
    result_list = [__get_failed_scoring_result_for_batch(request_obj, response_headers_empty=True)]

    # Act
    # Ensure output formatting is resilient to empty headers
    output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output(result_list, batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request
    for idx, result in enumerate(actual):
        output_obj = json.loads(result)
        assert output_obj["response"]["request_id"] is None
        assert output_obj["response"]["status_code"] == 400
        assert output_obj["error"]["message"]["error"]["type"] == "invalid_request_error"
        assert "maximum context length is 8190 tokens" in output_obj["error"]["message"]["error"]["message"]


@pytest.mark.parametrize("tiktoken_fails", [True, False])
@pytest.mark.parametrize("reorder_results", [True, False])
@pytest.mark.parametrize("input_schema_version", [1, 2])
def test_output_formatter_batch_20(
        monkeypatch,
        mock_get_logger,
        input_schema_version,
        reorder_results,
        tiktoken_fails):
    """Test convert result list batch size 20 case."""
    # Arrange
    batch_size_per_request = 20
    num_batches = 4
    full_batches = num_batches - 1
    additional_rows = 2

    result_list = []
    inputlists = []
    for n in range(full_batches):
        inputlist = __get_input_batch(batch_size_per_request)
        request_obj = {"input": inputlist, "custom_id": "task_123"}
        outputlist = __get_output_data(batch_size_per_request)
        result = __get_scoring_result_for_batch(
            batch_size_per_request,
            request_obj,
            outputlist,
            reorder_results,
            tiktoken_fails)
        inputlists.extend(inputlist)
        result_list.append(result)
    inputlist = __get_input_batch(additional_rows)
    request_obj = {"input": inputlist, "custom_id": "task_456"}
    outputlist = __get_output_data(additional_rows)

    inputlists.extend(inputlist)
    result_list.append(__get_scoring_result_for_batch(
        additional_rows,
        request_obj,
        outputlist,
        reorder_results,
        tiktoken_fails))

    # Act
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output(result_list, batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request * full_batches + additional_rows
    for idx, result in enumerate(actual):
        output_obj = json.loads(result)
        if input_schema_version == 1:
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

        response_obj = output_obj["response"] if input_schema_version == 1 else output_obj["response"]["body"]

        assert response_obj["data"][0]["index"] == valid_batch_idx
        assert response_obj["usage"]["total_tokens"] == valid_batch_len
        if tiktoken_fails:
            # Prompt tokens will equal total tokens (equals batch length)
            assert response_obj["usage"]["prompt_tokens"] == valid_batch_len
        else:
            # Batch pool case; prompt tokens is 10 + batch index (see helper function: `__get_token_counts`)
            assert response_obj["usage"]["prompt_tokens"] == valid_batch_idx + 10


@pytest.mark.parametrize("input_schema_version", [1, 2])
def test_incorrect_data_length_raises(input_schema_version):
    """Test incorrect data length raises."""
    # Arrange
    batch_size_per_request = 2
    result_list = []
    inputstring = __get_input_batch(batch_size_per_request)
    request_obj = {"input": inputstring, "custom_id": "task_123"}
    outputlist = __get_output_data(0)
    result = __get_scoring_result_for_batch(batch_size_per_request, request_obj, outputlist)
    result_list.append(result)

    # Act
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()

    with pytest.raises(Exception) as excinfo:
        output_formatter.format_output(result_list, batch_size_per_request)

    # Assert
    assert "Result data length 0 != 2 request batch length." in str(excinfo.value)


@pytest.mark.parametrize("input_schema_version", [1, 2])
def test_endpoint_response_is_not_json(input_schema_version, mock_get_logger):
    """Test endpoint response is not json."""
    # Arrange failed response payload as a string
    batch_size_per_request = 10
    inputs = __get_input_batch(batch_size_per_request)

    result = ScoringResult(
        status=ScoringResultStatus.FAILURE,
        start=0,
        end=0,
        model_response_code=400,
        request_metadata="Not important",
        response_headers={"header1": "value"},
        num_retries=2,
        token_counts=(1,) * batch_size_per_request,
        request_obj={"input": inputs, "custom_id": "task_123"},
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
    if input_schema_version == 1:
        output_formatter = V1OutputFormatter()
    else:
        output_formatter = V2OutputFormatter()
    actual = output_formatter.format_output([result], batch_size_per_request)

    # Assert
    assert len(actual) == batch_size_per_request
    for idx, result in enumerate(actual):
        unit = json.loads(result)
        if input_schema_version == 1:
            assert unit['request']['input'] == inputs[idx]
            assert type(unit['response']) is str
            assert 'maximum context length' in unit['response']
        elif input_schema_version == 2:
            assert unit['response']['request_id'] is not None
            assert unit['response']['status_code'] == 400
            assert type(unit['error']['message']) is str
            assert 'maximum context length' in unit['error']['message']


def __get_scoring_result_for_batch(batch_size, request_obj, outputlist, reorder_results=False, tiktoken_failed=False):
    token_counts = __get_token_counts(tiktoken_failed, request_obj["input"])
    result = ScoringResult(
        status=ScoringResultStatus.SUCCESS,
        model_response_code=200,
        start=0,
        end=0,
        request_metadata="Not important",
        response_headers={"header1": "value"},
        num_retries=2,
        token_counts=token_counts,
        request_obj=request_obj,
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


def __get_failed_scoring_result_for_batch(request_obj, tiktoken_failed=False, response_headers_empty=False):
    token_counts = __get_token_counts(tiktoken_failed, request_obj["input"])
    result = ScoringResult(
        status=ScoringResultStatus.FAILURE,
        start=0,
        end=0,
        model_response_code=400,
        request_metadata="Not important",
        response_headers=None if response_headers_empty else {"header1": "value"},
        num_retries=2,
        token_counts=token_counts,
        request_obj=request_obj,
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
