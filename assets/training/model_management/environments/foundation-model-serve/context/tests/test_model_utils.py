import json
import os
import shutil
import tempfile
import unittest

from model_utils import build_configs_from_model


class TestModelUtils(unittest.TestCase):
    def setUp(self):
        self.mlmodel = {
            "flavors":
                {
                    "hftransformersv2":{
                        "task_type": "text-generation"
                    }
                },
            "metadata": 
                {
                    "base_model_name": "Meta-Llama-3.1-8B",
                    "base_model_task": "text-generation",
                    "model_provider_name": "llama",
                    "is_common_api_enabled": True
                }
            }
        self.model_path = "mock_model_path"
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.inference_path = os.path.join(self.temp_dir, "inference_config.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_build_model_config(self):
        # TODO: Remove test once all models adopt inference_config.json
        config_content = {
            "architectures": ["LlamaForCausalLM"]
        }
        with open(self.config_path, 'w+') as config:
            json.dump(config_content, config)

        engine_config, _, _, task_type, model_info = build_configs_from_model(self.mlmodel, self.model_path, self.config_path, self.model_path, self.inference_path)

        expected_engine_name = "vllm"
        expected_task_type = "text-generation"
        self.assertEqual(engine_config["engine_name"], expected_engine_name)
        self.assertEqual(task_type, expected_task_type)
        self.assertEqual(model_info["is_common_api_enabled"], True)
        

    def test_build_model_config_inference_config(self):
        # Vllm Engine
        config_content = {
            "architectures": ["LlamaForCausalLM"]
        }
        inference_content = {
            "inference_engine": "mii"
        }
        with open(self.config_path, 'w+') as config:
            json.dump(config_content, config)

        with open(self.inference_path, "w+") as inference_config:
            json.dump(inference_content, inference_config)

        engine_config, _, _, task_type, _ = build_configs_from_model(self.mlmodel, self.model_path, self.config_path, self.model_path, self.inference_path)

        expected_engine_name = "mii"
        expected_task_type = "text-generation"
        self.assertEqual(engine_config["engine_name"], expected_engine_name)
        self.assertEqual(task_type, expected_task_type)