# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import unittest
from unittest.mock import patch, Mock, MagicMock
import os

from fastapi import HTTPException

from context.foundation.model.serve.api_server_setup.protocol import ChatRole, ChatMessage, ChatCompletionRequest, ContentPartType, ContentPart
from context.foundation.model.serve.request_adapter import BaseAdapter, VllmChatCompletionsAdapter, MixtralChatCompletionAdapter
from context.foundation.model.serve.constants import ExtraParameters


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

class TestRequestAdapter(unittest.TestCase):
    def test_base_adapter(self):
        messages = [
                ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
                ChatMessage(role=ChatRole.user, content="Hello, how are you?"),
        ]
        chat_request = {
            "messages": [message.model_dump() for message in messages]
        }
        
        raw_request_mock = MagicMock()
        raw_request_mock.headers = {
            ExtraParameters.KEY: None
        }
        
        req = BaseAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(chat_request, req.model_dump())

        chat_request["extra"] = "extra-param"
        with pytest.raises(HTTPException) as ex:
            BaseAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        assert "Extra parameters ['extra'] are not allowed" in ex.value.detail

        raw_request_mock.headers = {
            ExtraParameters.KEY: ExtraParameters.PASS_THROUGH
        }
        req = BaseAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(chat_request, req.model_dump())

        raw_request_mock.headers = {
            ExtraParameters.KEY: ExtraParameters.ERROR
        }
        with pytest.raises(HTTPException) as ex:
            BaseAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        assert "Extra parameters ['extra'] are not allowed" in ex.value.detail
    
    def test_vllm_chat_adapter(self):
        messages = [
                ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
                ChatMessage(role=ChatRole.user, content="Hello, how are you?"),
        ]
        chat_request = {
            "messages": [message.model_dump() for message in messages]
        }
        
        raw_request_mock = MagicMock()
        raw_request_mock.headers = {
            ExtraParameters.KEY: None
        }
        
        req = VllmChatCompletionsAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(chat_request, req.model_dump())

        chat_request["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "get_flight_info",
                    "description": "get_flight_info",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "loc_origin": {
                                "type": "string",
                                "description": "The departure airport, e.g. MIA"
                            },
                            "loc_destination": {
                                "type": "string",
                                "description": "The destination airport, e.g. NYC"
                            }
                        },
                        "required": ["loc_origin", "loc_destination"]
                    }
                }
            }
        ]
        with pytest.raises(HTTPException) as ex:
            VllmChatCompletionsAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        assert "Tools and tool_choice are not supported." in ex.value.detail

    def test_mixtral_chat_adapter(self):
        # [system, user]
        messages = [
                ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
                ChatMessage(role=ChatRole.user, content="Hello, how are you?"),
        ]

        expected_updated_msgs = [
            ChatMessage(role=ChatRole.user, content=[
                ContentPart(type=ContentPartType.text, text="You are a helpful assistant."),
                ContentPart(type=ContentPartType.text, text="Hello, how are you?")
            ])
        ]
        chat_request = {
            "messages": [message.model_dump() for message in messages]
        }
        
        expected_updated_req = {
            "messages": [message.model_dump() for message in expected_updated_msgs]
        }
        raw_request_mock = MagicMock()
        raw_request_mock.headers = {
            ExtraParameters.KEY: None
        }
        req = MixtralChatCompletionAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(expected_updated_req, req.model_dump())

        # [system, assistant]
        messages = [
                ChatMessage(role=ChatRole.system, content="You are a helpful assistant."),
                ChatMessage(role=ChatRole.assistant, content="Hello, how are you?"),
        ]

        expected_updated_msgs = [
                ChatMessage(role=ChatRole.user, content="You are a helpful assistant."),
                ChatMessage(role=ChatRole.assistant, content="Hello, how are you?"),
        ]
        chat_request = {
            "messages": [message.model_dump() for message in messages]
        }
        expected_updated_req = {
            "messages": [message.model_dump() for message in expected_updated_msgs]
        }
        req = MixtralChatCompletionAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(expected_updated_req, req.model_dump())

        chat_request["extra"] = "extra-param"
        with pytest.raises(HTTPException) as ex:
            MixtralChatCompletionAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        assert "Extra parameters ['extra'] are not allowed" in ex.value.detail

        expected_updated_req = {
            "messages": [message.model_dump() for message in expected_updated_msgs],
            "extra": "extra-param"
        }
        raw_request_mock.headers = {
            ExtraParameters.KEY: ExtraParameters.PASS_THROUGH
        }
        req = MixtralChatCompletionAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        compare_dict(expected_updated_req, req.model_dump())

        raw_request_mock.headers = {
            ExtraParameters.KEY: ExtraParameters.ERROR
        }
        with pytest.raises(HTTPException) as ex:
            MixtralChatCompletionAdapter(ChatCompletionRequest(**chat_request), raw_request_mock).adapt()
        assert "Extra parameters ['extra'] are not allowed" in ex.value.detail
        

if __name__ == "__main__":
    unittest.main()
