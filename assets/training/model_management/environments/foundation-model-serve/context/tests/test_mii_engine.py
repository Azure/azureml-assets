# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pytest
import unittest
from unittest.mock import patch, AsyncMock

from configs import EngineConfig, TaskConfig, MiiEngineConfig
from constants import TaskType
from engine.mii_engine import MiiEngine


class TestMiiEngine(unittest.TestCase):
    mii_config = MiiEngineConfig(deployment_name="test-deployment", mii_configs={})
    engine_config = EngineConfig(engine_name="mii",
                                 model_id="model-id",
                                 tokenizer="tokenizer",
                                 tensor_parallel=1,
                                 num_replicas=1,
                                 ml_model_info={},
                                 mii_config=mii_config
                                 )
    task_config = TaskConfig(task_type=TaskType.TEXT_GENERATION)
    chat_task_config = TaskConfig(task_type=TaskType.CONVERSATIONAL)

    @patch("engine.mii_engine.MiiEngine._file_restructure")
    @patch("engine.mii_engine.mii.MIIServer")
    def test_load_model(self, mock_mii_server, mock_file_restructure):
        mock_file_restructure.return_value = None
        mock_mii_server.return_value = None
        engine = MiiEngine(self.engine_config, self.task_config)
        engine.load_model()
        mock_mii_server.assert_called_once()

    @patch("engine.mii_engine.MiiEngine._file_restructure")
    @patch("engine.mii_engine.mii.MIIClient")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.is_port_open")
    def test_is_healthy(self, mock_is_port_open, mock_client, mock_file_restructure):
        mock_file_restructure.return_value = None
        mock_is_port_open.return_value = True
        mock_client.return_value = "model"

        engine = MiiEngine(self.engine_config, self.task_config)
        self.assertIsNone(engine.init_client())
        self.assertIsNotNone(engine.model)

    @patch("engine.mii_engine.MiiEngine._file_restructure")
    def test_get_tokens(self, mock_restructure):
        mock_restructure.return_value = None
        engine_config = EngineConfig(engine_name="mii-v1",
                                     model_id="hf-internal-testing/llama",
                                     tokenizer="hf-internal-testing/llama-tokenizer",
                                     tensor_parallel=1,
                                     num_replicas=1,
                                     ml_model_info={},
                                     mii_config=self.mii_config
                                     )
        engine = MiiEngine(engine_config, self.task_config)
        test_tokens = engine.get_tokens("This is a test. A token counting test. How many tokens will the llama count?")
        print(f"tokens counted: {len(test_tokens)}")
        self.assertIsNotNone(test_tokens)

    def test_file_restructure(self):
        model_path = os.path.join(os.getcwd(), "model")
        tokenizer_path = os.path.join(os.getcwd(), "tokenizer")

        directories = [model_path, tokenizer_path]
        files = ["tokenizer.json", "tokenizer.model"]

        for directory in directories:
            os.makedirs(directory)

        for file in files:
            file_path = os.path.join(tokenizer_path, file)
            with open(file_path, "x"):
                continue

        engine_config = EngineConfig(engine_name="mii-v1",
                                     model_id=model_path,
                                     tokenizer=tokenizer_path,
                                     tensor_parallel=1,
                                     num_replicas=1,
                                     ml_model_info={},
                                     mii_config=self.mii_config
                                     )
        MiiEngine(engine_config, self.task_config)

        for file in files:
            model_files = os.path.join(os.getcwd(), "model", file)
            tokenizer_files = os.path.join(os.getcwd(), "tokenizer", file)
            self.assertTrue(os.path.exists(model_files))
            self.assertTrue(os.path.exists(tokenizer_files))
            os.remove(model_files)
            os.remove(tokenizer_files)

        for directory in directories:
            os.rmdir(directory)


class TestMiiEngineAsync:
    mii_config = MiiEngineConfig(deployment_name="test-deployment", mii_configs={})
    engine_config = EngineConfig(engine_name="mii",
                                 model_id="model-id",
                                 tokenizer="tokenizer",
                                 tensor_parallel=1,
                                 num_replicas=1,
                                 ml_model_info={},
                                 mii_config=mii_config
                                 )
    task_config = TaskConfig(task_type=TaskType.TEXT_GENERATION)
    chat_task_config = TaskConfig(task_type=TaskType.CONVERSATIONAL)

    @patch("engine.mii_engine.MiiEngine._file_restructure")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    @pytest.mark.asyncio
    async def test_generate_text(self, mock_get_tokens, mock_file_restructure):
        mock_file_restructure.return_value = None
        mock_get_tokens.return_value = [1, 2, 3]

        engine = MiiEngine(self.engine_config, self.task_config)
        engine.model = AsyncMock()
        engine.model._request_async_response.return_value.response = ["response"]

        results = await engine.generate(["mock prompt"], {"max_length": 20})
        assert len(results) == 1
        assert results[0].response == "response"

    @patch("engine.mii_engine.MiiEngine._file_restructure")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    @pytest.mark.asyncio
    async def test_generate_chat(self, mock_get_tokens, mock_file_restructure):
        mock_file_restructure.return_value = None
        mock_get_tokens.return_value = [1, 2, 3]

        engine = MiiEngine(self.engine_config, self.chat_task_config)
        engine.model = AsyncMock()
        engine.model._request_async_response.return_value.response = ["promptresponse"]

        results = await engine.generate(["prompt"], {"max_length": 20})
        assert len(results) == 1
        assert results[0].response == "response"


if __name__ == "__main__":
    unittest.main()
