# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import unittest
from unittest.mock import patch

from engine.vllm_engine import VLLMEngine
from configs import EngineConfig, TaskConfig
from constants import TaskType


class TestVLLMEngine(unittest.TestCase):

    def setUp(self):
        self.engine_config = EngineConfig(engine_name="vllm",
                                          model_id="model_id",
                                          tokenizer="tokenizer",
                                          ml_model_info={}
                                        )
        self.task_config = TaskConfig(task_type=TaskType.TEXT_GENERATION)
        self.chat_task_config = TaskConfig(task_type=TaskType.CONVERSATIONAL)

    def tearDown(self):
        pass

    @patch("context.foundation.model.serve.engine.vllm_engine.subprocess.Popen")
    def test_load_model(self, mock_popen):
        engine = VLLMEngine(self.engine_config, self.task_config)
        engine.load_model()
        mock_popen.assert_called_once()

    @patch("context.foundation.model.serve.engine.vllm_engine.requests.post")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.is_port_open")
    def test_is_healthy(self, mock_is_port_open, mock_post):
        mock_is_port_open.return_value = True
        mock_post.return_value.status_code = 200
        engine = VLLMEngine(self.engine_config, self.task_config)
        self.assertIsNone(engine.init_client())

    @patch("context.foundation.model.serve.engine.vllm_engine.requests.post")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    def test_generate_text(self, mock_get_tokens, mock_post):
        mock_get_tokens.return_value = [1, 2, 3]
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = '{"text": ["response"]}' # this
        engine = VLLMEngine(self.engine_config, self.task_config)
        results = engine.generate(["prompt"], {"max_gen_len": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].response, "response")

    @patch("context.foundation.model.serve.engine.vllm_engine.requests.post")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    def test_generate_chat(self, mock_get_tokens, mock_post):
        mock_get_tokens.return_value = [1, 2, 3]
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = '{"text": ["promptresponse"]}' # this
        engine = VLLMEngine(self.engine_config, self.chat_task_config)
        results = engine.generate(["prompt"], {"max_gen_len": 10})
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].response, "response")

    @patch("context.foundation.model.serve.engine.vllm_engine.requests.post")
    @patch("context.foundation.model.serve.engine.engine.BaseEngine.get_tokens")
    def test_return_full_text(self, mock_get_tokens, mock_post):
        mock_get_tokens.return_value = [1, 2, 3]
        mock_post.return_value.status_code = 200
        mock_post.return_value.content = '{"text": ["promptresponse"]}'
        engine = VLLMEngine(self.engine_config, self.task_config)
        results = engine.generate(["prompt", "prompt"], {"max_gen_len": 10, "return_full_text": False})
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].response, "response")
        self.assertEqual(results[0].response, "response")

    def test_gen_params_to_vllm_params(self):
        engine = VLLMEngine(self.engine_config, self.task_config)
        params = {"max_gen_len": 10, "max_new_tokens": 5, "other_param": "value"}
        transformed_params = engine._gen_params_to_vllm_params(params)
        self.assertEqual(transformed_params, {"max_tokens": 5})

    def test_get_tokens(self):
        self.engine_config.model_id = "hf-internal-testing/llama"
        self.engine_config.tokenizer = "hf-internal-testing/llama-tokenizer"
        engine = VLLMEngine(self.engine_config, self.task_config)
        test_tokens = engine.get_tokens("This is a test. A token counting test. How many tokens will the llama count?")
        print(f"tokens counted: {len(test_tokens)}")
        self.assertIsNotNone(test_tokens)

    def test_verify_and_modify_tensor_parallel_size(self):
        # Llama, Codellama, Falcon-7b, and Falcon-40b attention heads
        attention_head_sizes = [32, 40, 71, 128]
        tensor_parallel_sizes = [i for i in range(1, 9)]
        expected_warnings = [(i, 71) for i in range(2, 9)]
        expected_warnings.extend([(3, 32), (5, 32), (6, 32), (7, 32), (3, 40),
                                  (6, 40), (7, 40), (3, 128), (5, 128), (6, 128), (7, 128)])
        for size in tensor_parallel_sizes:
            for head_size in attention_head_sizes:
                self.engine_config.tensor_parallel = size
                self.engine_config.model_config = {"num_attention_heads": head_size, 
                                                   "num_key_value_heads": head_size}

                if (size, head_size) in expected_warnings:
                    with self.assertLogs("engine.vllm_engine", level="WARNING") as logs:
                        _ = VLLMEngine(self.engine_config, self.task_config)
                        self.assertTrue(any(
                            "Tensor parallel size was incompatible" in output for output in logs.output
                        ))
                engine = VLLMEngine(self.engine_config, self.task_config)
                if (size, head_size) not in expected_warnings:
                    self.assertTrue(engine._vllm_kwargs["tensor-parallel-size"] == size)
                self.assertTrue(head_size % engine._vllm_kwargs["tensor-parallel-size"] == 0)

    def test_verify_and_modify_tensor_parallel_size_kv_size(self):
        tensor_parallel_sizes = [i for i in range(1, 9)]
        valid_sizes = [1, 2, 5]
        expected_warnings = [i for i in range(1, 9) if i not in valid_sizes]
        for size in tensor_parallel_sizes:
            self.engine_config.tensor_parallel = size
            self.engine_config.model_config = {"num_attention_heads": 40, 
                                                "num_key_value_heads": 10}
            if size in expected_warnings:
                with self.assertLogs("engine.vllm_engine", level="WARNING") as logs:
                        _ = VLLMEngine(self.engine_config, self.task_config)
                        self.assertTrue(any(
                            "Tensor parallel size was incompatible" in output for output in logs.output
                        ))
            engine = VLLMEngine(self.engine_config, self.task_config)
            if size not in expected_warnings:
                self.assertTrue(engine._vllm_kwargs["tensor-parallel-size"] == size)
            self.assertTrue(40 % engine._vllm_kwargs["tensor-parallel-size"] == 0)
            self.assertTrue(10 % engine._vllm_kwargs["tensor-parallel-size"] == 0)

    @patch.dict(os.environ, {
        "ENABLE_LORA": "True",
        "WORKER_USE_RAY": "False"
    })
    def test_user_args_set(self):
        self.engine_config.vllm_args = ["enable-lora"]
        self.engine_config.model_config = {"model_type": "phi3"}
        expected_args = {"trust-remote-code", "enable-lora", "disable-log-requests"}
        engine = VLLMEngine(self.engine_config, self.task_config)
        self.assertEqual(expected_args, set(engine._vllm_args))

    @patch.dict(os.environ, {
        "TRUST_REMOTE_CODE": "False",
        "ENABLE_LORA": "True",
        "WORKER_USE_RAY": "True",
        "DISABLE_LOG_REQUESTS": "False"
    })
    def test_user_args_override_defaults(self):
        self.engine_config.vllm_args = ["enable-lora", "worker-use-ray"]
        expected_args = {"enable-lora", "worker-use-ray"}
        self.engine_config.model_config = {"model_type": "phi3"}
        engine = VLLMEngine(self.engine_config, self.task_config)
        self.assertEqual(expected_args, set(engine._vllm_args))

if __name__ == "__main__":
    unittest.main()
