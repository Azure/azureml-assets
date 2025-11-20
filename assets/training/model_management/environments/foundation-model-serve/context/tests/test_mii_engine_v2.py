import pytest
import unittest
from unittest.mock import patch, AsyncMock

from configs import EngineConfig, TaskConfig, MiiEngineConfig
from constants import TaskType
from engine.mii_engine_v2 import MiiEngineV2


class MockResponse:
    def __init__(self, generated_text):
        self.generated_text = generated_text


class TestMiiEngineV2(unittest.TestCase):
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

    # TODO: fix test by mocking MIIServer correctly
    # @patch("engine.mii_engine_v2.mii.backend.server.MIIServer")
    # def test_load_model(self, mock_mii_server):
    #     mock_mii_server.return_value = None
    #     engine = MiiEngineV2(self.engine_config, self.task_config)
    #     engine.load_model()
    #     mock_mii_server.assert_called_once()

    @patch("engine.mii_engine_v2.mii.backend.client.MIIClient")
    @patch("foundation.model.serve.engine.engine.BaseEngine.is_port_open")
    def test_is_healthy(self, mock_is_port_open, mock_client):
        mock_is_port_open.return_value = True
        mock_client.return_value = "model"
        engine = MiiEngineV2(self.engine_config, self.task_config)
        self.assertIsNone(engine.init_client())
        self.assertIsNotNone(engine.model)

    def test_sampling_params(self):
        engine = MiiEngineV2(self.engine_config, self.task_config)
        params = {"max_gen_len": 10, "max_tokens": 5, "other_param": "value"}
        transformed_params = engine._gen_params_to_mii_params(params)
        self.assertEqual(transformed_params, {"max_new_tokens": 5})

    def test_get_tokens(self):
        engine_config = EngineConfig(engine_name="mii",
                                     model_id="hf-internal-testing/llama",
                                     tokenizer="hf-internal-testing/llama-tokenizer",
                                     tensor_parallel=1,
                                     num_replicas=1,
                                     ml_model_info={},
                                     mii_config=self.mii_config
                                    )
        engine = MiiEngineV2(engine_config, self.task_config)
        test_tokens = engine.get_tokens("This is a test. A token counting test. How many tokens will the llama count?")
        print(f"tokens counted: {len(test_tokens)}")
        self.assertIsNotNone(test_tokens)


class TestMiiEngineV2Async:
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

    @patch("foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    @pytest.mark.asyncio
    async def test_generate_text(self, mock_get_tokens):
        mock_get_tokens.return_value = [1, 2, 3]

        engine = MiiEngineV2(self.engine_config, self.task_config)
        engine.model = AsyncMock()
        engine.model._request_async_response.return_value = [MockResponse("response")]

        results = await engine.generate(["mock prompt"], {"max_length": 20})
        assert len(results) == 1
        assert results[0].response == "response"

    @patch("foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    @pytest.mark.asyncio
    async def test_generate_chat(self, mock_get_tokens):
        mock_get_tokens.return_value = [1, 2, 3]

        engine = MiiEngineV2(self.engine_config, self.chat_task_config)
        engine.model = AsyncMock()
        engine.model._request_async_response.return_value = [MockResponse("response")]

        results = await engine.generate(["mock prompt"], {"max_length": 20})
        assert len(results) == 1
        assert results[0].response == "response"


if __name__ == "__main__":
    unittest.main()
