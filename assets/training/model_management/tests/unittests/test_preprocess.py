# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test preprocess."""

import os
import json
import pytest
import unittest
import yaml
from azureml.model.mgmt.processors.factory import (
    get_mlflow_convertor,
    SupportedNLPTasks,
    SupportedVisionTasks,
    SupportedDiffusersTask,
    SupportedTasks,
)
from azureml.model.mgmt.processors.transformers.convertors import HFMLFLowConvertor, NLPMLflowConvertor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


@pytest.fixture
def config():
    """Test model config."""
    return {
        "architectures": ["BertForMaskedLM"],
        "attention_probs_dropout_prob": 0.1,
        "gradient_checkpointing": False,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "transformers_version": "4.6.0.dev0",
        "type_vocab_size": 2,
        "use_cache": True,
        "vocab_size": 28996,
    }


@pytest.fixture
def tokenizer_config():
    """Test tokenizer config."""
    return {"do_lower_case": False}


@pytest.fixture
def tokenizer():
    """Test Tokenizer."""
    return {
        "version": "1.0",
        "truncation": "null",
        "padding": "null",
        "added_tokens": [
            {
                "id": 0,
                "special": True,
                "content": "[PAD]",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
            },
        ],
        "normalizer": {
            "type": "BertNormalizer",
        },
        "pre_tokenizer": {
            "type": "BertPreTokenizer",
        },
        "post_processor": {
            "type": "TemplateProcessing",
        },
        "decoder": {
            "type": "WordPiece",
        },
        "model": {
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
        },
    }


@pytest.fixture
def translate_params():
    """Test translate params used in hf MLflow conversion."""
    return {"task": "fill-mask", "model_id": "bert-base-cased"}


@pytest.fixture
def vocab():
    """Test vocab."""
    return """##ieri
comparisons
forensic
186
Giro
skeptical
disciplinary
battleship
"""


@pytest.fixture
def model_path(config, tokenizer, tokenizer_config, vocab):
    """Create an input model path based on test configs."""
    config_file_name = "config.json"
    tokenizer_file_name = "tokenizer.json"
    tokenizer_config_file_name = "tokenizer_config.json"
    vocab_file_name = "vocab.txt"
    pytorch_model_file_name = "pytorch_model.bin"

    json_file_details = {
        config_file_name: config,
        tokenizer_file_name: tokenizer,
        tokenizer_config_file_name: tokenizer_config,
    }

    txt_file_details = {vocab_file_name: vocab, pytorch_model_file_name: "Model"}

    with TemporaryDirectory() as working_dir:
        for file_name, details in json_file_details.items():
            file_path = os.path.join(working_dir, file_name)
            with open(file_path, "w") as f:
                json.dump(details, f)

        for file_name, details in txt_file_details.items():
            file_path = os.path.join(working_dir, file_name)
            Path(file_path).write_text(details)
        yield working_dir


class TestFactoryModule(unittest.TestCase):
    """Test HF Model Convertor Factory."""

    @patch("azureml.model.mgmt.processors.factory.NLPMLflowConvertorFactory")
    def test_get_nlp_mlflow_convertor(self, mock_nlp_factory):
        """Test NLP model MLflow convertor."""
        model_framework = "Huggingface"
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedNLPTasks.FILL_MASK.value}
        mock_convertor = mock_nlp_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_nlp_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.factory.VisionMLflowConvertorFactory")
    def test_get_vision_mlflow_convertor(self, mock_vision_factory):
        """Test vision model MLflow convertor."""
        model_framework = "Huggingface"
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedVisionTasks.IMAGE_CLASSIFICATION.value}
        mock_convertor = mock_vision_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_vision_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.factory.DiffusersMLflowConvertorFactory")
    def test_get_diffusers_mlflow_convertor(self, mock_diffusers_factory):
        """Test diffusers model MLflow convertor."""
        model_framework = "Huggingface"
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedDiffusersTask.TEXT_TO_IMAGE.value}
        mock_convertor = mock_diffusers_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_diffusers_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.factory.ASRMLflowConvertorFactory")
    def test_get_asr_mlflow_convertor(self, mock_asr_factory):
        """Test asr model MLflow convertor."""
        model_framework = "Huggingface"
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value}
        mock_convertor = mock_asr_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_asr_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    def test_get_mlflow_convertor_unsupported_task(self):
        """Test unsupported task case."""
        model_framework = "Huggingface"
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"
        translate_params = {"task": "unsupported_task"}

        with self.assertRaises(Exception) as context:
            get_mlflow_convertor(model_framework, model_dir, output_dir, temp_dir, translate_params)

        self.assertTrue("unsupported_task" in str(context.exception))


class TestHFMLFLowConvertor:
    """Test HF Model Convertor Factory."""

    def test_mlflow_conda_dep(self, model_path, translate_params):
        """Validate conda dep has needed dependencies."""
        with TemporaryDirectory() as output_dir, TemporaryDirectory() as temp_dir:
            # save model
            nlp_mlflow_convertor = NLPMLflowConvertor(
                model_dir=model_path, output_dir=output_dir, temp_dir=temp_dir, translate_params=translate_params
            )
            nlp_mlflow_convertor = nlp_mlflow_convertor.save_as_mlflow()

            # validate pycocotools
            conda_file_path = os.path.join(output_dir, HFMLFLowConvertor.CONDA_FILE_NAME)
            with open(conda_file_path) as f:
                conda_dict = yaml.safe_load(f)
            conda_deps = conda_dict["dependencies"]
            assert "pycocotools=2.0.4" in conda_deps
