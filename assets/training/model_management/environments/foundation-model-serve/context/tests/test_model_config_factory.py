import pytest
from unittest.mock import patch, MagicMock
from model_config_factory import ModelConfigFactory
from constants import TaskType


@patch("model_config_factory.DiffusionConfigurationBuilder", new_callable=MagicMock())
def test_get_model_config_for_stable_diffusion(mock_diffusion_config):
    task = TaskType.TEXT_TO_IMAGE
    model_type = "stable-diffusion"

    config = ModelConfigFactory.get_config_builder(task, model_type=model_type)

    mock_diffusion_config.assert_called_once_with(task)
    assert config is mock_diffusion_config.return_value


@patch("model_config_factory.DiffusionConfigurationBuilder", new_callable=MagicMock())
def test_get_model_config_for_stable_diffusion_inpainting(mock_diffusion_config):
    task = TaskType.TEXT_TO_IMAGE_INPAINTING
    model_type = "stable-diffusion"

    config = ModelConfigFactory.get_config_builder(task, model_type=model_type)

    mock_diffusion_config.assert_called_once_with(task)
    assert config is mock_diffusion_config.return_value
