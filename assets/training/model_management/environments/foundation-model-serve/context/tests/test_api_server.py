<<<<<<< HEAD
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model package containing model serving functionality."""
import asyncio
import json
import unittest
from unittest.mock import patch, MagicMock
from engine import InferenceResult
from configs import EngineConfig
from fastapi.testclient import TestClient
from api_server import app, filter_swagger_paths_by_tag
from context.foundation.model.serve.constants import SupportedTask, ModelInfo
from context.foundation.model.serve.api_server_setup.protocol import ChatRole, ChatMessage, UsageInfo
from dataclasses import asdict


def compare_dict(obj, cmp):
    for k, v in obj.items():
        if isinstance(v, dict):
            compare_dict(v, cmp[k])
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    compare_dict(item, cmp[k][i])
                elif isinstance(item, list):
                    compare_list(item, cmp[k][i])
                else:
                    assert item == cmp[k][i]
        else:
            assert v == cmp[k]


def compare_list(lis, cmp):
    for i, item in enumerate(lis):
        if isinstance(item, dict):
            compare_dict(item, cmp[i])
        elif isinstance(lis, list):
            for i, item in enumerate(lis):
                if isinstance(item, dict):
                    compare_dict(item, cmp[i][i])
                elif isinstance(item, list):
                    compare_list(item, cmp[i][i])
                else:
                    assert item == cmp[i]
        else:
            assert item == cmp[i]


def mock_model_info():
    return {
        ModelInfo.MODEL_NAME: "meme",
        ModelInfo.MODEL_TYPE: "text-generation",
        ModelInfo.MODEL_PROVIDER: "meme"
    }


def get_serving_url(task_type):
    if task_type == SupportedTask.CHAT_COMPLETION:
        return f"/chat/completions"
    elif task_type == SupportedTask.TEXT_GENERATION:
        return f"/completions"
    else:
        return ""


def compare_stream_response_and_chunks(chunks, stream_response):
    cmp = []
    for line in stream_response.iter_lines():
        if line:
            # line should be bytes but somehow is string, so pretend it should be string for now
            # decoded_line = line.decode('utf-8')
            decoded_line = line
            try:
                decoded_data = decoded_line.split("data: ")[1]
                decoded_json = json.loads(decoded_data)
                cmp.append(decoded_json)
            except Exception as e:
                print(f"Error parsing decoded line: {decoded_line}, with exception: {e}")

    for i, chunk in enumerate(chunks):
        compare_dict(chunk, cmp[i])


class TestAPIServer(unittest.TestCase):
    client = TestClient(app)
    messages = [
        ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.user, content="Hello, how are you?"),
    ]
    chat_request = {
        "messages": [message.model_dump() for message in messages],
        "model": "default-ignored",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["User:", "Assistant:"],
        "stream": False,
    }
    textgen_request = {
        "model": "default-ignored",
        "prompt": "my favorite color is blue because",
        "temperature": 0.7
    }
    vllm_chat_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                    "role": "assistant"
                }
            }
        ],
        "created": 1677664795,
        "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        "model": "vllm-returned-model",
        "object": "chat.completion",
        "usage": UsageInfo(
            completion_tokens=16,
            prompt_tokens=10,
            total_tokens=26
        ).model_dump(),
    }
    vllm_textgen_response = {
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "text": "\n\n\"Let Your Sweet Tooth Run Wild at Our Creamy Ice Cream Shack"
            }
        ],
        "created": 1683130927,
        "id": "cmpl-7C9Wxi9Du4j1lQjdjhxBlO22M61LD",
        "model": "vllm-returned-model",
        "object": "text_completion",
        "usage": UsageInfo(
            completion_tokens=37,
            prompt_tokens=10,
            total_tokens=47
        ).model_dump(),
    }
    inference_results = [InferenceResult("This is the response", 1.0, 1.0, ["token"], 1)]
    vllm_stream_textgen_response_chunks = [{'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': '\n',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': '\n',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': 'An',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': ' LL',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': 'M',
                                                         'finish_reason': 'length'}],
                                            'usage': {'prompt_tokens': 7,
                                                      'total_tokens': 12,
                                                      'completion_tokens': 5}},
                                           ]
    vllm_stream_chat_response_chunks = [{'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'role': 'assistant'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' The'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' answer'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' to'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' the'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' mathematical'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' expression'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' '},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': '2'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' +'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' '},
                                                      'finish_reason': 'length'}],
                                         'usage': {'prompt_tokens': 32,
                                                   'total_tokens': 42,
                                                   'completion_tokens': 10}},
                                        ]

    def mock_sse_stream_iter_lines(self, chunks):
        for chunk in chunks:
            json_string = json.dumps(chunk)
            sse_formatted_line = f"data: {json_string}\n\n"
            yield sse_formatted_line.encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')

    def test_health(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def async_return(self, result):
        f = asyncio.Future()
        return f

    def test_swagger_paths_filter_by_tag(self):
        target_tag = "target_tag"
        sample_openapi_schema = {
            "paths": {
                "p1": {"op1": {"tags": [target_tag]}},
                "p2": {"op1": {}},
                "p3": {"op1": {"tags": ["not_target"]}},
                "p4": {"op1": {"tags": [target_tag, "not_target"]},
                       "op2": {"tags": ["anything"]}}
            }
        }

        filtered = filter_swagger_paths_by_tag(sample_openapi_schema, target_tag)
        assert (set(filtered["paths"].keys()) == set(["p1", "p2", "p4"]))
        assert (set(filtered["paths"]["p4"].keys()) == set(["op1"]))

        filtered = filter_swagger_paths_by_tag(sample_openapi_schema, "")
        assert (set(filtered["paths"].keys()) == set(["p2"]))

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_chat_completion_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.json.return_value = self.vllm_chat_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        final_response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=self.chat_request)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_chat_response = self.vllm_chat_response.copy()
        final_chat_response['model'] = expected_model
        compare_dict(final_chat_response, final_response.json())

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_fmscorer')
    def test_chat_completion_missing_messages(self, mock_fmscorer):
        request_data = {
            "temperature": 0.8
        }
        response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=request_data)
        assert response.status_code == 422
        response_json = {"error": {"code": "Invalid input", "status": 422, "message": "invalid input error", "details": [
            {"type": "missing", "loc": ["body", "messages"], "msg": "Field required", "input": {"temperature": 0.8}}]}}
        compare_dict(response.json(), response_json)

        request_data["messages"] = []
        response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=request_data)
        response_json = {"error": {"code": "Invalid input",
                                   "status": 422,
                                   "message": "invalid input error",
                                   "details": [{"type": "value_error",
                                                "loc": ["body",
                                                        "messages"],
                                                "msg": "Value error, messages can not be an empty list",
                                                "input": [],
                                                "ctx": {"error": {}}}]}}
        assert response.status_code == 422
        compare_dict(response.json(), response_json)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_api_version_and_extra_parameters(self, mock_fmscorer):
        request_data = {
            "model": "gpt-3.5-turbo",
            "prompt": "my favorite color is blue because",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        # mock_served_model.return_value = 'g_served_model'
        # mock_engine_config.return_value = EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")

        request_data["some-random-param"] = "hello"
        response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=request_data)
        assert response.status_code == 400
        assert f"Extra parameters ['some-random-param'] are not allowed" in response.json()["detail"]

        response = self.client.post(
            get_serving_url(
                SupportedTask.TEXT_GENERATION),
            json=request_data,
            headers={
                "extra-parameters": "not-exist"})
        assert response.status_code == 400
        assert f"Unexpected EXTRA_PARAMETERS option" in response.json()["detail"]

        response = MagicMock()
        response.json.return_value = self.vllm_textgen_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        response = self.client.post(
            get_serving_url(
                SupportedTask.TEXT_GENERATION),
            json=request_data,
            headers={
                "extra-parameters": "pass-through"})
        assert response.status_code == 200
        compare_dict(self.vllm_textgen_response, response.json())

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_text_generation_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.json.return_value = self.vllm_textgen_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=self.textgen_request)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_textgen_response = self.vllm_textgen_response.copy()
        final_textgen_response['model'] = expected_model
        compare_dict(final_textgen_response, final_response.json())

    # [TODO] prepare an invalid test for openai api server case

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_fmscorer')
    def test_text_generation_invalid_request(self, mock_fmscorer):
        request_data = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=request_data)
        assert final_response.status_code == 422
        final_response_json = final_response.json()
        response_json = {
            "error": {
                "code": "Invalid input",
                "status": 422,
                "message": "invalid input error",
                "details": [
                    {
                        "type": "missing",
                        "loc": [
                            "body",
                            "prompt"],
                        "msg": "Field required",
                        "input": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "max_tokens": 100}}]}}
        compare_dict(final_response_json, response_json)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_stream_text_generation_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.iter_lines = MagicMock(
            side_effect=lambda: self.mock_sse_stream_iter_lines(
                self.vllm_stream_textgen_response_chunks))
        response.status_code = 200
        response.headers = {
            "Content-Type": "application/json"
        }
        mock_fmscorer.run_openai_async.return_value = response
        textgen_request_stream = self.textgen_request.copy()
        textgen_request_stream['stream'] = True
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=textgen_request_stream)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_stream_textgen_response_chunks = self.vllm_stream_textgen_response_chunks.copy()
        for chunk in final_stream_textgen_response_chunks:
            chunk.update({"model": expected_model})
        compare_stream_response_and_chunks(final_stream_textgen_response_chunks, final_response)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_stream_chat_completion_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.iter_lines = MagicMock(
            side_effect=lambda: self.mock_sse_stream_iter_lines(
                self.vllm_stream_chat_response_chunks))
        response.headers = {
            "Content-Type": "application/json"
        }
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        chat_request_stream = self.chat_request.copy()
        chat_request_stream['stream'] = True
        final_response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=chat_request_stream)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_stream_chat_response_chunks = self.vllm_stream_chat_response_chunks.copy()
        for chunk in final_stream_chat_response_chunks:
            chunk.update({"model": expected_model})
        # print stream response chunks
        for line in final_response.iter_lines():
            if line:
                print(line)
        compare_stream_response_and_chunks(final_stream_chat_response_chunks, final_response)


if __name__ == "__main__":
    unittest.main()
=======
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Model package containing model serving functionality."""
import asyncio
import json
import unittest
import pytest
from unittest.mock import patch, MagicMock
from engine import InferenceResult
from configs import EngineConfig
from fastapi.testclient import TestClient
from api_server import app, filter_swagger_paths_by_tag
import api_server
from context.foundation.model.serve.constants import SupportedTask, ModelInfo
from context.foundation.model.serve.api_server_setup.protocol import ChatRole, ChatMessage, UsageInfo
from dataclasses import asdict


def compare_dict(obj, cmp):
    for k, v in obj.items():
        if isinstance(v, dict):
            compare_dict(v, cmp[k])
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    compare_dict(item, cmp[k][i])
                elif isinstance(item, list):
                    compare_list(item, cmp[k][i])
                else:
                    assert item == cmp[k][i]
        else:
            assert v == cmp[k]


def compare_list(lis, cmp):
    for i, item in enumerate(lis):
        if isinstance(item, dict):
            compare_dict(item, cmp[i])
        elif isinstance(lis, list):
            for i, item in enumerate(lis):
                if isinstance(item, dict):
                    compare_dict(item, cmp[i][i])
                elif isinstance(item, list):
                    compare_list(item, cmp[i][i])
                else:
                    assert item == cmp[i]
        else:
            assert item == cmp[i]


def mock_model_info():
    return {
        ModelInfo.MODEL_NAME: "meme",
        ModelInfo.MODEL_TYPE: "text-generation",
        ModelInfo.MODEL_PROVIDER: "meme"
    }


def get_serving_url(task_type):
    if task_type == SupportedTask.CHAT_COMPLETION:
        return f"/chat/completions"
    elif task_type == SupportedTask.TEXT_GENERATION:
        return f"/completions"
    else:
        return ""


def compare_stream_response_and_chunks(chunks, stream_response):
    cmp = []
    for line in stream_response.iter_lines():
        if line:
            # line should be bytes but somehow is string, so pretend it should be string for now
            # decoded_line = line.decode('utf-8')
            decoded_line = line
            try:
                decoded_data = decoded_line.split("data: ")[1]
                decoded_json = json.loads(decoded_data)
                cmp.append(decoded_json)
            except Exception as e:
                print(f"Error parsing decoded line: {decoded_line}, with exception: {e}")

    for i, chunk in enumerate(chunks):
        compare_dict(chunk, cmp[i])


class TestAPIServer(unittest.TestCase):
    client = TestClient(app)
    messages = [
        ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
        ChatMessage(role=ChatRole.user, content="Hello, how are you?"),
    ]
    chat_request = {
        "messages": [message.model_dump() for message in messages],
        "model": "default-ignored",
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 100,
        "stop": ["User:", "Assistant:"],
        "stream": False,
    }
    textgen_request = {
        "model": "default-ignored",
        "prompt": "my favorite color is blue because",
        "temperature": 0.7
    }
    vllm_chat_response = {
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
                    "role": "assistant"
                }
            }
        ],
        "created": 1677664795,
        "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
        "model": "vllm-returned-model",
        "object": "chat.completion",
        "usage": UsageInfo(
            completion_tokens=16,
            prompt_tokens=10,
            total_tokens=26
        ).model_dump(),
    }
    vllm_textgen_response = {
        "choices": [
            {
                "finish_reason": "length",
                "index": 0,
                "text": "\n\n\"Let Your Sweet Tooth Run Wild at Our Creamy Ice Cream Shack"
            }
        ],
        "created": 1683130927,
        "id": "cmpl-7C9Wxi9Du4j1lQjdjhxBlO22M61LD",
        "model": "vllm-returned-model",
        "object": "text_completion",
        "usage": UsageInfo(
            completion_tokens=37,
            prompt_tokens=10,
            total_tokens=47
        ).model_dump(),
    }
    inference_results = [InferenceResult("This is the response", 1.0, 1.0, ["token"], 1)]
    vllm_stream_textgen_response_chunks = [{'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': '\n',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': '\n',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': 'An',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': ' LL',
                                                         'finish_reason': None}],
                                            'usage': None},
                                           {'id': 'cmpl-b306f98794fc4a2496147f11304045f4',
                                            'created': 1715647995,
                                            'model': '/data/mlflow_model_folder/data/model',
                                            'choices': [{'index': 0,
                                                         'text': 'M',
                                                         'finish_reason': 'length'}],
                                            'usage': {'prompt_tokens': 7,
                                                      'total_tokens': 12,
                                                      'completion_tokens': 5}},
                                           ]
    vllm_stream_chat_response_chunks = [{'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'role': 'assistant'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' The'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' answer'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' to'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' the'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' mathematical'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' expression'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' '},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': '2'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' +'},
                                                      'finish_reason': None}]},
                                        {'id': 'cmpl-7c77e37de3b446c3a70806d7ddaf3754',
                                         'object': 'chat.completion.chunk',
                                         'created': 1715653435,
                                         'model': '/data/mlflow_model_folder/data/model',
                                         'choices': [{'index': 0,
                                                      'delta': {'content': ' '},
                                                      'finish_reason': 'length'}],
                                         'usage': {'prompt_tokens': 32,
                                                   'total_tokens': 42,
                                                   'completion_tokens': 10}},
                                        ]

    def mock_sse_stream_iter_lines(self, chunks):
        for chunk in chunks:
            json_string = json.dumps(chunk)
            sse_formatted_line = f"data: {json_string}\n\n"
            yield sse_formatted_line.encode('utf-8')
        yield "data: [DONE]\n\n".encode('utf-8')

    def test_health(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def async_return(self, result):
        f = asyncio.Future()
        return f

    def test_swagger_paths_filter_by_tag(self):
        target_tag = "target_tag"
        sample_openapi_schema = {
            "paths": {
                "p1": {"op1": {"tags": [target_tag]}},
                "p2": {"op1": {}},
                "p3": {"op1": {"tags": ["not_target"]}},
                "p4": {"op1": {"tags": [target_tag, "not_target"]},
                       "op2": {"tags": ["anything"]}}
            }
        }

        filtered = filter_swagger_paths_by_tag(sample_openapi_schema, target_tag)
        assert (set(filtered["paths"].keys()) == set(["p1", "p2", "p4"]))
        assert (set(filtered["paths"]["p4"].keys()) == set(["op1"]))

        filtered = filter_swagger_paths_by_tag(sample_openapi_schema, "")
        assert (set(filtered["paths"].keys()) == set(["p2"]))

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_chat_completion_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.json.return_value = self.vllm_chat_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        final_response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=self.chat_request)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_chat_response = self.vllm_chat_response.copy()
        final_chat_response['model'] = expected_model
        compare_dict(final_chat_response, final_response.json())

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_fmscorer')
    def test_chat_completion_missing_messages(self, mock_fmscorer):
        request_data = {
            "temperature": 0.8
        }
        response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=request_data)
        assert response.status_code == 422
        response_json = {"error": {"code": "Invalid input", "status": 422, "message": "invalid input error", "details": [
            {"type": "missing", "loc": ["body", "messages"], "msg": "Field required", "input": {"temperature": 0.8}}]}}
        compare_dict(response.json(), response_json)

        request_data["messages"] = []
        response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=request_data)
        response_json = {"error": {"code": "Invalid input",
                                   "status": 422,
                                   "message": "invalid input error",
                                   "details": [{"type": "value_error",
                                                "loc": ["body",
                                                        "messages"],
                                                "msg": "Value error, messages can not be an empty list",
                                                "input": [],
                                                "ctx": {"error": {}}}]}}
        assert response.status_code == 422
        compare_dict(response.json(), response_json)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_api_version_and_extra_parameters(self, mock_fmscorer):
        request_data = {
            "model": "gpt-3.5-turbo",
            "prompt": "my favorite color is blue because",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        # mock_served_model.return_value = 'g_served_model'
        # mock_engine_config.return_value = EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")

        request_data["some-random-param"] = "hello"
        response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=request_data)
        assert response.status_code == 400
        assert f"Extra parameters ['some-random-param'] are not allowed" in response.json()["detail"]

        response = self.client.post(
            get_serving_url(
                SupportedTask.TEXT_GENERATION),
            json=request_data,
            headers={
                "extra-parameters": "not-exist"})
        assert response.status_code == 400
        assert f"Unexpected EXTRA_PARAMETERS option" in response.json()["detail"]

        response = MagicMock()
        response.json.return_value = self.vllm_textgen_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        response = self.client.post(
            get_serving_url(
                SupportedTask.TEXT_GENERATION),
            json=request_data,
            headers={
                "extra-parameters": "pass-through"})
        assert response.status_code == 200
        compare_dict(self.vllm_textgen_response, response.json())

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_text_generation_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.json.return_value = self.vllm_textgen_response
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=self.textgen_request)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_textgen_response = self.vllm_textgen_response.copy()
        final_textgen_response['model'] = expected_model
        compare_dict(final_textgen_response, final_response.json())

    # [TODO] prepare an invalid test for openai api server case

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_fmscorer')
    def test_text_generation_invalid_request(self, mock_fmscorer):
        request_data = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.8,
            "top_p": 0.9,
            "max_tokens": 100,
        }
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=request_data)
        assert final_response.status_code == 422
        final_response_json = final_response.json()
        response_json = {
            "error": {
                "code": "Invalid input",
                "status": 422,
                "message": "invalid input error",
                "details": [
                    {
                        "type": "missing",
                        "loc": [
                            "body",
                            "prompt"],
                        "msg": "Field required",
                        "input": {
                            "model": "gpt-3.5-turbo",
                            "temperature": 0.8,
                            "top_p": 0.9,
                            "max_tokens": 100}}]}}
        compare_dict(final_response_json, response_json)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.TEXT_GENERATION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_stream_text_generation_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.iter_lines = MagicMock(
            side_effect=lambda: self.mock_sse_stream_iter_lines(
                self.vllm_stream_textgen_response_chunks))
        response.status_code = 200
        response.headers = {
            "Content-Type": "application/json"
        }
        mock_fmscorer.run_openai_async.return_value = response
        textgen_request_stream = self.textgen_request.copy()
        textgen_request_stream['stream'] = True
        final_response = self.client.post(get_serving_url(SupportedTask.TEXT_GENERATION), json=textgen_request_stream)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_stream_textgen_response_chunks = self.vllm_stream_textgen_response_chunks.copy()
        for chunk in final_stream_textgen_response_chunks:
            chunk.update({"model": expected_model})
        compare_stream_response_and_chunks(final_stream_textgen_response_chunks, final_response)

    @patch('api_server.g_served_model', new="g_served_model")
    @patch('api_server.task_type', new=SupportedTask.CHAT_COMPLETION)
    @patch('api_server.g_engine_config', new=asdict(EngineConfig(engine_name="vllm", model_id="model_id", tokenizer="tokenizer")))
    @patch('api_server.g_model_info', new=mock_model_info())
    @patch('api_server.g_fmscorer')
    def test_stream_chat_completion_valid_request(self, mock_fmscorer):
        response = MagicMock()
        response.iter_lines = MagicMock(
            side_effect=lambda: self.mock_sse_stream_iter_lines(
                self.vllm_stream_chat_response_chunks))
        response.headers = {
            "Content-Type": "application/json"
        }
        response.status_code = 200
        mock_fmscorer.run_openai_async.return_value = response
        chat_request_stream = self.chat_request.copy()
        chat_request_stream['stream'] = True
        final_response = self.client.post(get_serving_url(SupportedTask.CHAT_COMPLETION), json=chat_request_stream)
        assert final_response.status_code == 200
        expected_model = "meme"
        final_stream_chat_response_chunks = self.vllm_stream_chat_response_chunks.copy()
        for chunk in final_stream_chat_response_chunks:
            chunk.update({"model": expected_model})
        # print stream response chunks
        for line in final_response.iter_lines():
            if line:
                print(line)
        compare_stream_response_and_chunks(final_stream_chat_response_chunks, final_response)


if __name__ == "__main__":
    unittest.main()
>>>>>>> 4736c86812f5a79482f8001ee49abe9393309f85
