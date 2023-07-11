# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test preprocess."""

import unittest
from unittest.mock import patch
from azureml.model.mgmt.processors.transformers.factory import (
    get_mlflow_convertor,
    SupportedNLPTasks,
    SupportedVisionTasks,
    SupportedDiffusersTask,
    SupportedTasks,
)


class TestFactoryModule(unittest.TestCase):
    """Test HF Model Convertor Factory."""

    @patch("azureml.model.mgmt.processors.transformers.factory.NLPMLflowConvertorFactory")
    def test_get_nlp_mlflow_convertor(self, mock_nlp_factory):
        """Test NLP model MLflow convertor."""
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedNLPTasks.FILL_MASK.value}
        mock_convertor = mock_nlp_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_nlp_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.transformers.factory.VisionMLflowConvertorFactory")
    def test_get_vision_mlflow_convertor(self, mock_vision_factory):
        """Test vision model MLflow convertor."""
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedVisionTasks.IMAGE_CLASSIFICATION.value}
        mock_convertor = mock_vision_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_vision_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.transformers.factory.DiffusersMLflowConvertorFactory")
    def test_get_diffusers_mlflow_convertor(self, mock_diffusers_factory):
        """Test diffusers model MLflow convertor."""
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedDiffusersTask.TEXT_TO_IMAGE.value}
        mock_convertor = mock_diffusers_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_diffusers_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    @patch("azureml.model.mgmt.processors.transformers.factory.ASRMLflowConvertorFactory")
    def test_get_asr_mlflow_convertor(self, mock_asr_factory):
        """Test asr model MLflow convertor."""
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"

        translate_params = {"task": SupportedTasks.AUTOMATIC_SPEECH_RECOGNITION.value}
        mock_convertor = mock_asr_factory.create_mlflow_convertor.return_value
        result = get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)
        self.assertEqual(result, mock_convertor)
        mock_asr_factory.create_mlflow_convertor.assert_called_once_with(
            model_dir,
            output_dir,
            temp_dir,
            translate_params,
        )

    def test_get_mlflow_convertor_unsupported_task(self):
        """Test unsupported task case."""
        model_dir = "/path/to/model_dir"
        output_dir = "/path/to/output_dir"
        temp_dir = "/path/to/temp_dir"
        translate_params = {"task": "unsupported_task"}

        with self.assertRaises(Exception) as context:
            get_mlflow_convertor(model_dir, output_dir, temp_dir, translate_params)

        self.assertTrue("unsupported_task" in str(context.exception))
