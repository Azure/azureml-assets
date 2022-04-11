"""
Tests running the pytorch_image_classifier/model/ loader
"""
import os
import sys
import tempfile
import pytest
from unittest.mock import patch

import torch

from components.pytorch_image_classifier.model import MODEL_ARCH_LIST, get_model_metadata, load_model

# IMPORTANT: see conftest.py for fixtures

# we only care about patching those specific mlflow methods
@pytest.mark.parametrize("model_arch", MODEL_ARCH_LIST)
def test_model_loader(model_arch):
    """Tests src/components/pytorch_image_classifier/model/"""
    model_metadata = get_model_metadata(model_arch)

    assert model_metadata is not None
    assert isinstance(model_metadata, dict)
    assert "library" in model_metadata
    assert "input_size" in model_metadata

    # using pretrained=False to avoid downloading each time we unit test
    model = load_model(model_arch, output_dimension=4, pretrained=False)

    assert model is not None
    assert isinstance(model, torch.nn.Module)
